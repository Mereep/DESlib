"""
# Copyright 2018 Professorship Media Informatics, University of Applied Sciences Mittweida
# licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# @author Richard Vogel, 
# @email: richard.vogel@hs-mittweida.de
# @created: 06.10.2019
"""
import numpy as np
import math
from numpy.random import RandomState
from abc import ABCMeta, abstractmethod
from sklearn.utils.validation import check_is_fitted, check_random_state, check_X_y
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import BaggingClassifier
from typing import Optional, List, Union, Dict, Callable
import logging
import time
import datetime
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance_matrix
from sklearn.tree import DecisionTreeClassifier
from .utils import BaseGonStepCallBack


class CustomDSBase(BaseEstimator, ClassifierMixin):
    """
    Base Model for classifiers that build their during training process
    """

    # if no classifier is given thats the amount of
    # standard created classifiers
    _default_pool_size: int = 10

    def __init__(self,
                 pool_classifiers: Optional[Union[List[BaseEstimator], BaseEstimator]] = None,
                 DSEL_perc: Optional[float] = None,
                 random_state: Optional[Union[int, RandomState]] = None,
                 needs_proba: bool = False):

        self.random_state = random_state
        self.pool_classifiers = pool_classifiers
        self.needs_proba = needs_proba
        self.DSEL_perc = DSEL_perc
        self.pool_classifiers_: Optional[List[BaseEstimator]] = pool_classifiers
        self.random_state_: Optional[RandomState] = None
        self.n_classifiers_: Optional[int] = None

        self.classes_: Optional[List[str]] = None
        self.enc_: Optional[LabelEncoder] = None
        self.n_classes_: Optional[int] = None
        self.n_features_: Optional[int] = None
        self.n_samples_: Optional[int] = None
        self._is_fitted: bool = False

        self.train_target_: Optional[np.ndarray] = None
        self.train_data_: Optional[np.ndarray] = None
        self.train_data_processed_: Optional[np.ndarray] = None
        self.train_data_processed_target__: Optional[np.ndarray] = None

        self.DSEL_data_: Optional[np.ndarray] = None
        self.DSEL_processed_: Optional[np.ndarray] = None
        self.DSEL_target_: Optional[np.ndarray] = None
        self.DSEL_processed_target_: Optional[np.ndarray] = None

        self.scaler_: Optional[StandardScaler] = None

    def reset(self):
        """
        Will put the classifier into an unfit status
        by reseting all variables ending with _
        :return:
        """
        for key, value in self.__dict__.items():
            if key[-1] == '_':
                if len(key) > 1 and key[:-2] != '__':
                    self.__setattr__(key, None)

        self._is_fitted = False

    def score(self, X, y, sample_weight=None):
        y = self._preprocess_y(y)
        return super().score(X, y, sample_weight)

    def estimate_competence(self,
                            query):
        """
        Parameters
        ----------
        query : array of shape  = [n_samples, n_features]
            The test examples.


        Returns
        -------
        competences : array of shape = [n_samples, n_classifiers]
            Competence level estimated for each base classifier and test
            example.
        """
        assert self.is_fitted(), "The model has to be fitted to " \
                                 "estimate competence (Code: 2384023)"

    @abstractmethod
    def select(self, query: np.ndarray):
        """
        Should select the base classifier given a query
        :return:
        """
        pass

    def fit(self, X, y):
        """
        Fit the classifier to the given data.

        Parameters
        ----------
        :param X: Training AND DSEL data (if DSEL_perc is None or 1. thats the same)

        :param y: Class labels

        :raises ValueError: if classifiers are not supported
        :return:
        """
        self.reset()
        self.random_state_ = check_random_state(self.random_state)

        # Check if the length of X and y are consistent.
        X, y = check_X_y(X, y)

        # Check if the pool of classifiers is None or a BaseEstimator.
        # If yes, use a BaggingClassifier for the pool.
        if not isinstance(self.pool_classifiers, List):
            if len(X) < 2:
                raise ValueError('More than one sample is needed '
                                 'if the pool of classifiers is not informed.')

            if not isinstance(self.pool_classifiers, BaseEstimator):
                self.pool_classifiers = None

            # Split the dataset into training (for the base classifier) and
            # DSEL (for DS)

            self.pool_classifiers_ = [DecisionTreeClassifier(random_state=self.random_state,
                                                             max_depth=4)
                                      for i in range(self._default_pool_size)]

        else:
            self.pool_classifiers_ = self.pool_classifiers

        if not self._check_base_classifiers_prefit():
            raise ValueError("At least one of your models is not a classifier "
                             "or needs a predict_proba method")

        # if we want to split the train and DSEL set
        if self.DSEL_perc is None or self.DSEL_perc == 1.:
            X_dsel = X.__copy__()
            y_dsel = y.__copy__()
        else:
            X, X_dsel, y, y_dsel = train_test_split(
                X, y, test_size=self.DSEL_perc,
                random_state=self.random_state_, stratify=y)

        self.n_classifiers_ = len(self.pool_classifiers_)
        self._setup_label_encoder(y)
        self._set_train_data(X, y)
        self._set_dsel(X_dsel, y_dsel)

        self.fit_ensemble()
        self._is_fitted = True

    @abstractmethod
    def fit_ensemble(self):
        """
        This method implements the main algorithm
        :return:
        """
        pass

    def _set_train_data(self, X, y):
        """Pre-Process the input X and y data into the dynamic selection
        dataset(DSEL) and get information about the structure of the data
        (e.g., n_classes, n_samples, n_features)

        Parameters
        ----------
        X : array of shape = [n_samples, n_features]
            The Input data.

        y : array of shape = [n_samples]
            class labels of each sample in X.
        """

        self.n_classes_ = self.classes_.shape[0]
        self.n_features_ = X.shape[1]
        self.n_samples_ = X.shape[0]

        self.train_data_ = X
        self.train_target_ = y
        self.train_data_processed_ = self._preprocess_X(X=X)
        self.train_data_processed_target = self._preprocess_y(y=y)

        self.train_data_ = self._preprocess_X(X)
        self.train_data_processed_target_ = self._preprocess_y(y)

    def _set_dsel(self, X, y):

        self.scaler_ = StandardScaler()
        self.scaler_.fit(X)

        self.dsel_std_ = X.std(axis=0) + 10e-5
        self.dsel_max = X.max(axis=0)
        self.dsel_min = X.min(axis=0)

        self.DSEL_data_ = X
        self.DSEL_target_ = y

        self.DSEL_processed_ = self._preprocess_dsel(X=X)
        self.DSEL_processed_target_ = self._preprocess_y(y=y)

    def _preprocess_dsel(self, X: np.ndarray) -> np.ndarray:
        """
        Preprocesses the data for use with DSEL

        :param X:
        :return:
        """
        return self.scaler_.transform(X)

    def _preprocess_X(self, X: np.ndarray):
        """
        Will preprocess the X (feature data)
        :return:
        """
        return X

    def _preprocess_y(self, y: np.ndarray):
        """
        Will preprocess labels

        :param y:
        :return:
        """
        return self.enc_.transform(y)

    def _check_base_classifiers_prefit(self) -> bool:
        """
        Check sanity of base classifiers BEFORE fit
        :return:
        """

        probabilistic_ok = not self.needs_proba or np.all([hasattr(clf, 'predict_proba')
                                                           for clf in self.pool_classifiers_])

        is_classifier = np.all(isinstance(clf, ClassifierMixin) for clf in self.pool_classifiers_)

        return probabilistic_ok and is_classifier

    def _check_base_classifier_fitted(self):
        """ Checks if each base classifier in the pool is fitted.

        Raises
        -------
        NotFittedError: If any of the base classifiers is not yet fitted.
        """
        for clf in self.pool_classifiers:
            check_is_fitted(clf, "classes_")

    def _setup_label_encoder(self, y):
        self.enc_ = LabelEncoder()
        self.enc_.fit(y)
        self.classes_ = self.enc_.classes_

    def is_fitted(self) -> bool:
        """
        Checks if the model is fit
        :return:
        """
        return self._is_fitted

    def log(self, level: int, msg: str):
        """
        Logs a message
        :param level:
        :param msg:
        :return:
        """
        # print(msg)
        logging.getLogger(self.__class__.__name__).log(level, msg)

    def log_info(self, msg):
        self.log(level=logging.INFO, msg=msg)

    def log_warning(self, msg):
        self.log(level=logging.WARNING, msg=msg)

    def log_debug(self, msg):
        self.log(level=logging.DEBUG, msg=msg)

    @abstractmethod
    def predict(self, X: np.ndarray) -> float:
        pass


class CustomDCSBase(CustomDSBase):
    pass


class GON(CustomDCSBase):
    """Gang of Nerds Model (GON).

    Will construct


    Parameters
    ----------

    :param step size: determines the movement speed
    :param iterations: amount of cycles
    :param fixed_classifiers: List of len(pool_classifiers)
    :param assign_random_points_if_model_outside: force some random points to be assigned to the model even if its
    :param stochastic_attention: If < 1 will use only a subset of DSEL data to validate against each iteration
    outside current assignments area (Prevent dying models)
    where and entry of True means the classifier will NOT be refit during process

    References
    ----------

    """

    def __init__(self,
                 pool_classifiers: Optional[Union[List[BaseEstimator], BaseEstimator]] = None,
                 step_size: float = 3,
                 iterations: int = 20,
                 fixed_classifiers: Optional[List[int]] = None,
                 assign_random_points_if_model_outside: bool = True,
                 stochastic_attention: float = 1.,
                 step_callback: Optional[BaseGonStepCallBack] = None,
                 mode: str = 'global',
                 *args,
                 **kwargs,
                 ):

        super().__init__(pool_classifiers=pool_classifiers,
                         *args,
                         **kwargs)

        self._step_size = step_size
        self._iterations = iterations
        self._assign_random_points_if_model_outside = assign_random_points_if_model_outside
        self.model_positions_: Optional[np.ndarray] = None
        self._step_callback = step_callback

        assert self._iterations >= 1, "You have to iterate at least 1 round (Code: 237498237)"
        assert self._step_size > 0, "Step size has to be greater than zero (Code: 823742893)"
        assert mode in ('local', 'global', 'mixed'), "Mode has to be either local, global or mixed (Code: 9996573482)"
        assert stochastic_attention > 0. and stochastic_attention <= 1., f"Stochastic attention level " \
                                                                         f"has to be between " \
                                                                         f"]0, 1]. (Code: 93284209348)"
        self._mode = mode
        self._stochastic_attention = stochastic_attention

        # if classifiers are not marked fixed or not fixed we mark them as non-fixed (all of them)
        self._fixed_classifiers = fixed_classifiers if fixed_classifiers is not None \
            else [False for i in range(self._default_pool_size \
                                           if not self.pool_classifiers else len(self.pool_classifiers))]

        # Check some preconditions

        if self.pool_classifiers is not None:
            assert len(self._fixed_classifiers) == len(self.pool_classifiers), f"If you define fixed classifiers " \
                                                                               f"you have to provide that state for each " \
                                                                               f"classifier (#{len(self.pool_classifiers)}) " \
                                                                               "(Code: 2398420398)"

        if np.any(self._fixed_classifiers):
            if not self.pool_classifiers:
                raise ValueError(f"You have to provide fitted classifiers if you want to fix them (Code: 8789234563")
            for i, clf in enumerate(self.pool_classifiers):
                if self._fixed_classifiers[i] is True and not 'classes_' in dir(clf):
                    raise ValueError(f"Classifier index {i+1} is marked fixed but is not fitted (Code: 23742893)")

    def estimate_competence(self,
                            query):

        super().estimate_competence(query)

        X = np.atleast_2d(query)
        competences = np.zeros(shape=(len(query), len(self.get_current_classifiers())),
                               dtype=np.bool)

        assignments = self.assign_data_points_to_model(X, is_processed=False)

        for model_idx, assignment in enumerate(assignments):
            competences[assignment, model_idx] = 1

        return competences

    def _preprocess_dsel(self, X: np.ndarray) -> np.ndarray:
        return X

    def step(self, step_size: float):
        """
        Do one step (Pulling + fitting in that order)

        :param step_size:
        :return:
        """
        self._pull_classifiers_to_competence_area(step_size=step_size)
        self._fit_classifiers_to_closest_points()

    def fit_ensemble(self):
        start_time = datetime.datetime.now()

        self.model_positions_ = self._init_model_positions()
        self._fit_classifiers_to_closest_points()
        best_performance = self._calculate_performance()
        best_pos = self.model_positions_.copy()
        self.log_info(f"Start performance: {best_performance:.3f}")
        if self._mode == 'mixed':
            mixed_mode = True
            self._mode = 'global'
        else:
            mixed_mode = False
        for i in range(self._iterations):
            if mixed_mode is True and i == self._iterations // 2:
                self.log_info("Switched from global model to local mode (Fine-tuning)")

            round_time = datetime.datetime.now()
            self.log_info(f"Performing round {i}")

            step_size = self._decay_fn(i, self._iterations) * self._step_size

            self.log_info(f"Current step size: {step_size:.3f}")

            self.step(step_size=step_size)
            current_performance = self._calculate_performance()

            self.log_info(f"Performance: {current_performance:.3f}")

            if current_performance > best_performance:
                self.log_info(f"Performance increase by {(current_performance-best_performance):.3f}")
                best_performance = current_performance
                best_pos = self.model_positions_.__copy__()

            time_delta = datetime.datetime.now() - round_time

            if self._step_callback is not None:
                self._step_callback(iteration=i,
                                    model=self,
                                    performance_train=current_performance)

            self.log_info(f"Round took {time_delta.total_seconds():.3f} seconds\n")

            # if current_performance == 1.:
            #    self.log_info(f"Stopping due to performance is 1")
            #    break

        # restore best model positions
        self.model_positions_ = best_pos
        self._fit_classifiers_to_closest_points()
        # recalculate performance
        performance = self._calculate_performance()
        train_delta = datetime.datetime.now() - start_time
        self.log_info(f"Performance Final: {performance:.3f}")
        self.log_info(f"Training took {train_delta.total_seconds():.3f} seconds in sum")

    def _decay_fn(self, current_iteration: int, max_iteration: int) -> float:
        """
        Will produce a factor that can be used as decay for iteratively
        reduced parameters
        :param current_iteration:
        :param max_iteration:
        :return:
        """
        return math.exp(-current_iteration / max_iteration)

    def _pull_classifiers_to_competence_area(self, step_size: float = 0.8):
        """
        Will use DSEL to pull models to working clusters
        :return:
        """
        models = self.get_current_classifiers()
        dsel_data = self.get_DSEL(processed=True)
        dsel_labels = self.get_DSEL_target(processed=True)
        model_positions = self.model_positions_

        # only take some points into account if requested (stochastic attention)
        if self._stochastic_attention < 1.:
            # we leave the "draw with replacement"-option turned on
            # to reduce calculation time here
            data_indices = np.random.choice(np.arange(0, len(dsel_data), 1),
                                            int(self._stochastic_attention * len(dsel_data)))
            data_to_evaluate = dsel_data[data_indices]
            dsel_labels = dsel_labels[data_indices]
        else:
            # otherwise we use whole DSEL
            data_to_evaluate = dsel_data

        # local mode only predicts the already assigned neighbourhood
        if self._mode == 'local':
            assignments = self.assign_data_points_to_model(dsel_data, is_processed=True)
        else:
            assignments = np.arange(0, len(data_to_evaluate), 1)

        for i, model in enumerate(models):
            pos = model_positions[i]

            if self._mode == 'local':
                assignment = assignments[i]

                # pick random points in local mode if model is outside
                # this case cannot happen in global mode (since it will always predict ALL samples)
                if len(assignment) == 0 and self._assign_random_points_if_model_outside:
                    assignment = np.random.randint(0,
                                              len(dsel_data),
                                              max(2,
                                                  int(len(dsel_data) / len(models))))

            else:
                assignment = assignments

            # models eventually can die in local mode if we don't force them some point (which is optional)
            if len(assignment) > 0:
                pred = model.predict(data_to_evaluate[assignment])
                correct_predict_idx = np.where(pred == dsel_labels[assignment])[0]

                correct_predict_dsel = dsel_data[assignment][correct_predict_idx]

                if len(correct_predict_idx) > 0:
                    force = np.sum(correct_predict_dsel, axis=0) / len(correct_predict_idx)
                else:
                    force = np.zeros_like(pos)
                direction_to_force = force - pos

                direction_to_force *= step_size
                self.model_positions_[i] += direction_to_force

    def _calculate_performance(self) -> float:
        """
        Will calculate the performance in range 0..1
        using current configuration
        :return:
        """
        res = self.predict(self.get_train_data(processed=False))

        return accuracy_score(y_true=self.get_train_targets(), y_pred=res)

    def predict(self, X: any) -> np.ndarray:
        """
        Predicts using the current model

        :param X:
        :param prepare_data:
        :return:
        """
        assignments = self.assign_data_points_to_model(X=X)
        models = self.pool_classifiers_
        X_p = self._preprocess_X(X=X)
        predictions = np.ndarray(shape=(len(X),),
                                 dtype=self.get_DSEL_target(processed=True).dtype)

        # predict each point by its closest model
        for model_idx, data_points in assignments.items():
            model = models[model_idx]
            to_predict = X_p[data_points]

            if len(to_predict) > 0:
                predictions[data_points] = model.predict(X=to_predict)

        return predictions

    def get_current_classifiers(self):
        """
        Returns the current models inside
        :return:
        """
        return self.pool_classifiers_

    def _fit_classifiers_to_closest_points(self):
        """
        Will find closest models for each point and fits
        model on closest points

        :param take_random_points_if_model_outside: if no points assigned to model -> take random points ?
        """
        train_data_prepared = self.get_train_data(processed=True)
        train_targets_prepared = self.get_train_targets(processed=True)

        models = self.get_current_classifiers()
        model_to_points = self.assign_data_points_to_model(X=train_data_prepared, is_processed=True)

        for model_idx, assigned_data_indices in model_to_points.items():
            # if all data points are assigned somewhere else
            # we optionally pick some random points
            if len(assigned_data_indices) == 0 and self._assign_random_points_if_model_outside:
                assigned_data_indices = np.random.randint(0,
                                                          len(train_data_prepared),
                                                          max(2, int(len(train_data_prepared) / len(model_to_points))))
            try:
                models[model_idx].fit(train_data_prepared[assigned_data_indices],
                                      train_targets_prepared[assigned_data_indices])
            except Exception as e:
                self.log_warning(f"Could not fit model with idx {model_idx} due to {e.__str__()}. "
                                 f"I Leave it alone atm (Code: 739792658)")

    def assign_data_points_to_model(self, X: np.ndarray = None,
                                    is_processed: bool = False) -> Dict[int, np.ndarray]:
        """
        Returns a dict that maps model index to closest data points
        :return:
        """

        # transform data to DSEL format
        if X is not None:
            X_dsel = self._preprocess_dsel(X=X) if not is_processed else X
        else:
            X_dsel = self.get_DSEL(processed=True)

        # read model points
        model_points = self.model_positions_

        # create a samples x models distance matrix
        distances = np.zeros(shape=(X_dsel.shape[0],
                                    model_points.shape[0]))

        for model_idx in range(model_points.shape[0]):
            # p=1 minekowski distance normalized by stdev
            model_point = model_points[model_idx, :]
            distance = np.sum(np.abs((X_dsel - model_point)) / self.dsel_std_,
                              axis=1)

            distances[:, model_idx] = distance

        # get row-wise  minimal distance
        closest_models = np.argmin(distances, axis=1)

        # construct distionary {model: [data_indices]
        d = {idx: np.argwhere(closest_models == idx)[:, 0] for idx in range(len(model_points))}

        return d

    def get_DSEL(self, processed=True) -> np.ndarray:
        """
        Will return the DSEL data

        :param processed:
        :return:
        """
        assert self.DSEL_processed_ is not None, "DSEL data is not yet been set (propably " \
                                                 "the model is not fit) (Code; 328472389)"
        if processed:
            return self.DSEL_processed_

        return self.DSEL_data_

    def get_DSEL_target(self, processed=True) -> np.ndarray:
        """
        Will return the DSEL target

        :param processed:
        :return:
        """
        assert self.DSEL_processed_ is not None, "DSEL data is not yet been set (propably " \
                                                 "the model is not fit) (Code; 4564564564)"
        if processed:
            return self.DSEL_processed_target_

        return self.DSEL_target_

    def get_train_targets(self, processed=True) -> np.ndarray:
        """
          Will return the training targets / labels
          :return:
        """
        assert self.train_target_ is not None, "Training datas' labels is not yet been set (propably " \
                                               "the model is not fit) (Code; 4564564567)"

        if processed:
            return self.train_data_processed_target_

        return self.train_target_

    def get_train_data(self, processed=True) -> np.ndarray:
        """
        Will return the training data
        :param processed:
        :return:
        """
        assert self.train_data_processed_ is not None, "Training data data is not yet been set (propably " \
                                                       "the model is not fit) (Code; 4564564567)"
        if processed:
            return self.train_data_processed_
        else:
            return self.train_data_

    def get_data_dimensionality(self) -> int:
        """
        Gets the dimensionality of the data
        :return:
        """
        return self.get_DSEL().shape[1]

    def _init_model_positions(self) -> np.ndarray:
        """
        Will return initial points for the models in feature space
        :return:
        """
        pos = np.zeros(shape=(len(self.pool_classifiers_), self.get_data_dimensionality()), dtype=np.float)
        for i in range(len(self.get_current_classifiers())):
            pos[i, :] = [np.random.normal(loc=0,
                                          size=1,
                                          scale=self.dsel_std_[dim]) for dim in range(self.get_data_dimensionality())]
        return pos

        for dim in range(self.get_data_dimensionality()):
            pos[:, dim] = np.random.rand(pos.shape[0]) * abs(self.dsel_min[dim]) + abs(self.dsel_max[dim]) - \
                          self.dsel_min[dim]

        return pos
        # return np.random.rand(len(self.pool_classifiers_), self.get_data_dimensionality()) * \
        #           self.dsel_max - self.dsel_min

    def select(self, query: np.ndarray):
        super().select(query=query)
        assignments = self.assign_data_points_to_model(X=query, is_processed=False)
        selected_classifiers = np.ndarray(shape=(len(query),), dtype=np.object)
        for model_idx, assignment in enumerate(assignments):
            assignments[assignments] = self.get_current_classifiers()[model_idx]

        return selected_classifiers
