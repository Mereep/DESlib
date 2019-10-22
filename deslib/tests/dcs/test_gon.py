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
# @created: 20.10.2019
"""

import unittest
from deslib.dcs.gon import GON
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
import numpy as np
from deslib.dcs.utils import ScatterVideoCreatorGON
import matplotlib.pyplot as plt
import tempfile
from os.path import join as pjoin
import os


class GonTest(unittest.TestCase):

    def test_video(self):
        scatter = ScatterVideoCreatorGON(plot_images_directly=False)
        # should perform at 1 accuracy (4 model)
        models = [DecisionTreeClassifier(max_depth=1) for i in range(8)]
        t = GON(pool_classifiers=models,
                step_size=2.,
                iterations=40,
                DSEL_perc=0.2,
                step_callback=scatter)

        t.fit(self.X_dummy, self.y_dummy)

        scatter.as_video(fp=self.video_tmp,
                        animation=scatter.draw_animation())

        self.assertTrue(os.path.isfile(self.video_tmp))
        self.assertAlmostEqual(os.path.getsize(self.video_tmp), 8134, delta=200)


    def test_model(self):

        # should perform at 2/8 accuracy (1 model)
        models = [DecisionTreeClassifier(max_depth=1) for i in range(1)]
        t = GON(pool_classifiers=models,
                DSEL_perc=0.2)

        t.fit(self.X_dummy, self.y_dummy)
        self.assertEqual(t.score(self.X_dummy, self.y_dummy), 0.25)

        # should perform at 1/2 accuracy (2 model)
        models = [DecisionTreeClassifier(max_depth=1) for i in range(2)]
        t = GON(pool_classifiers=models,
                DSEL_perc=0.2)

        t.fit(self.X_dummy, self.y_dummy)
        self.assertEqual(t.score(self.X_dummy, self.y_dummy), 0.5)

        # should perform at 6/8 accuracy (3 model)
        models = [DecisionTreeClassifier(max_depth=1) for i in range(3)]
        t = GON(pool_classifiers=models,
                DSEL_perc=0.2,
                step_size=0.3)

        t.fit(self.X_dummy, self.y_dummy)
        self.assertEqual(t.score(self.X_dummy, self.y_dummy), 0.75)

        # should perform at 1 accuracy (4 model)
        models = [DecisionTreeClassifier(max_depth=1) for i in range(4)]
        t = GON(pool_classifiers=models,
                step_size=3.,
                iterations=100,
                DSEL_perc=0.3)

        t.fit(self.X_dummy, self.y_dummy)
        self.assertEqual(t.score(self.X_dummy, self.y_dummy), 1.)

    def test_assignments(self):
        """
        Checks if models
        :return:
        """
        models = [DecisionTreeClassifier(max_depth=4) for i in range(2)]

        t = GON(pool_classifiers=models,
                DSEL_perc=0.2)

        t.fit(self.X_dummy, self.y_dummy)
        assigments = t.assign_data_points_to_model(X=self.X_dummy)

        self.assertEqual(len(assigments), 2)

        s = 0
        assigned_indices = []
        for idx, assignment in assigments.items():
            s += len(assignment)
            assigned_indices += list(assignment)

        # all points assigned
        self.assertEqual(len(self.X_dummy), s)

        # ... and none of them twice
        self.assertEqual(set(assigned_indices), set(range(len(self.X_dummy))))

    def test_dsel_split(self):
        """
        Will check if split is proportionally ok
        :return:
        """
        models = [DecisionTreeClassifier(max_depth=4) for i in range(2)]

        t = GON(pool_classifiers=models,
                DSEL_perc=0.2)


        t.fit(self.X_dummy, self.y_dummy)

        self.assertEqual(len(t.get_DSEL()), 0.2 * len(self.X_dummy))
        self.assertEqual(len(t.get_train_data()), 0.8 * len(self.X_dummy))

    def test_no_classifiers_case(self):
        """
        Check auto generation of classifiers
        :return:
        """
        t = GON()
        t.fit(self.X_dummy, self.y_dummy)
        self.assertEqual(len(t.get_current_classifiers()), t._default_pool_size)

    def test_fixed_classifiers_preconditions(self):
        """
        Tests fixed classifier behaviour prechcecks
        :return:
        """

        # should raise due to standard classifiers are not fit
        with self.assertRaises(ValueError):
            t = GON(fixed_classifiers=[True for i in range(GON._default_pool_size)])

        pool = [DecisionTreeClassifier()]
        pool[0].fit(self.X_dummy, self.y_dummy)

        # should work since classifier is fixed
        t = GON(pool_classifiers=pool, fixed_classifiers=[True])

        # should raise due to classifier is not fit
        with self.assertRaises(ValueError):
            t = GON(fixed_classifiers=[True], pool_classifiers=[DecisionTreeClassifier()])

    def setUp(self):
        np.random.seed(0)
        self.X_dummy = np.reshape(np.array([
            np.random.normal(size=(100, 3), scale=0.5) + (0, 0, 0),
            np.random.normal(size=(100, 3), scale=0.5) + (0, 0, 10),
            np.random.normal(size=(100, 3), scale=0.5) + (0, 10, 0),
            np.random.normal(size=(100, 3), scale=0.5) + (0, 10, 10),
            np.random.normal(size=(100, 3), scale=0.5) + (10, 0, 0),
            np.random.normal(size=(100, 3), scale=0.5) + (10, 0, 10),
            np.random.normal(size=(100, 3), scale=0.5) + (10, 10, 0),
            np.random.normal(size=(100, 3), scale=0.5) + (10, 10, 10),
        ]), newshape=(-1, 3))

        self.y_dummy = np.reshape(np.array([np.ones(100) * i for i in range(8)]), -1)

        self.tmpdir = tempfile.gettempdir()
        self.video_tmp = pjoin(self.tmpdir, 'test_vid.htm')
        if os.path.isfile(self.video_tmp):
            os.remove(self.video_tmp)

    def tearDown(self):
        if os.path.isfile(self.video_tmp):
            os.remove(self.video_tmp)

