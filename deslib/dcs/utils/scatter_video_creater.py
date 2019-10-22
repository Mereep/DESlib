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
# @created: 21.10.2019
"""
import pandas as pd
from sklearn.decomposition import PCA
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import numpy as np


class BaseGonStepCallBack(ABC):

    @abstractmethod
    def __call__(self,
                 iteration: int,
                 model,
                 performance_train: float = None,
                 ax=None):
        pass


class ScatterVideoCreatorGON:

    def __init__(self, plot_images_directly: bool=False):

        self.data = None
        self.pca_model = None
        self.data_to_plot = None
        self.imgs = []
        self.model_point_history = []
        self.title_history = []
        self.plot_images_directly = plot_images_directly
        self.performance_history = []

    def __call__(self,
                 iteration: int,
                 model,
                 performance_train: float = None,
                 ax=None):
        """
        Gets called from tge model itself
        """
        if self.data is None:
            # prepare first data stuff
            self.data = model.get_DSEL(processed=True)

            # if we do not have 2 dimensions we use PCA to bring it down to 2
            if self.data.shape[1] != 2:
                self.pca_model = PCA(n_components=2)
                self.pca_model.fit(self.data)

            else:
                # otherwise we leave it untouched
                class PCADummy:
                    def transform(self, X):
                        return X
                self.pca_model = PCADummy()
            pca_data_points = self.pca_model.transform(self.data)

            self.data_to_plot = pd.DataFrame(pca_data_points, columns=['dim1', 'dim2'])
            self.data_to_plot.loc[:, 'label'] = model.get_DSEL_target(processed=True)


        # res = sns.scatterplot(x='dim1', y='dim2', hue='label', data=self.data_to_plot, ax=ax)
        # plt.sca(ax)
        # scatters = ax.scatter(self.data_to_plot['dim1'], self.data_to_plot['dim2'])

        model_scatters = []
        for i, expert in enumerate(model.get_current_classifiers()):

            coords = model.model_positions_[i]
            pcaed = self.pca_model.transform([coords])[0]
            model_point = (pcaed[0], pcaed[1])
            model_scatters.append(model_point)

        if self.plot_images_directly is True:
            if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))

            self.draw_scene(ax=ax,
                            model_points=model_scatters,
                            title=f"Round {iteration+1}, Optimization: {iteration+1}")

        # scatters.append(model_scatters)
        self.model_point_history.append(model_scatters)
        self.performance_history.append((performance_train))

    def draw_scene(self,
                   ax,
                   model_points,
                   draw_data_points: bool=True,
                   title: str = "",
                   performance_train: float=None,
                   performance_val: float=None):

        if ax is not None:
            plt.sca(ax)

        if draw_data_points:
            scatters = plt.scatter(self.data_to_plot['dim1'],
                                   self.data_to_plot['dim2'],
                                   c=self.data_to_plot['label'].values.astype(np.int32),
                                   animated=True)
        else:
            scatters = None

        model_scatters = []

        for i, model_point in enumerate(model_points):
            model_scatters += plt.plot(model_point[0],
                                       model_point[1],
                                       'o',
                                       markersize=20,
                                       color='red', animated=True)

            model_scatters += [plt.text(model_point[0],
                                        model_point[1],
                                        i + 1, animated=True)]

        title_drawer = plt.title(title)
        self.title_history.append(title)

        if scatters is not None:
            return [scatters, title_drawer, *model_scatters]
        else:
            return [title_drawer, *model_scatters]

    def draw_animation(self, interval: int=1000, repeat_delay: int=3000, title: str="") -> ArtistAnimation:
        """
        Will generate the animation itself
        use as_video if you want to save
        :param interval:
        :param repeat_delay:
        :return:
        """
        fig = plt.figure()

        scenes = []
        scatter_data_points = None
        for i, model_points in enumerate(self.model_point_history):
            ax = plt.gca()
            scene = self.draw_scene(ax=None,
                                    model_points=model_points,
                                    draw_data_points=i == 0,
                                    title="")
            round_text = plt.gca().text(0.04, 0.92,
                                        f"Round {(i+1).__str__()}/{len(self.model_point_history)}",
                                        #horizontalalignment="center",
                                        transform=ax.transAxes,
                                        bbox=dict(boxstyle="round",
                                                  ec=(1., 0.5, 0.5),
                                                  fc=(1., 0.8, 0.8),
                                                  )
                                        )
            performance_text = plt.gca().text(0.04, 0.05,
                                        f"Performance train: {round(self.performance_history[i], 2)}",
                                              #horizontalalignment="left"
                                              transform=ax.transAxes,
                                              bbox=dict(boxstyle="round",
                                                        ec=(1., 0.5, 0.5),
                                                        fc=(1., 0.8, 0.8),
                                                        )

                                              )
            if i == 0:
                scatter_data_points = scene[0]
                scenes.append([*scene, round_text, performance_text])
            else:
                scenes.append([scatter_data_points, *scene, round_text, performance_text])

        ani = ArtistAnimation(fig,
                              scenes,
                              interval=interval,
                              repeat_delay=repeat_delay)
        plt.title(title)
        return ani

    def as_video(self, fp: str, animation: ArtistAnimation):
        """
        Will save animation as video to fp
        :param fp:
        :return:
        """
        animation.save(fp)