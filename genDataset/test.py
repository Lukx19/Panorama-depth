import unittest
from glob import glob
import colapse_planes as cp
from utils import read_tiff, onehottify
import numpy as np


class TestColapsePlanes(unittest.TestCase):

    def testHottifySynthetic(self):
        mask = np.array([[0, 1], [2, 2], [3, 4]])
        res = onehottify(mask)
        self.assertEqual(res.shape, (3, 2, 5))
        res0 = np.transpose(res, (2, 0, 1)).tolist()
        self.assertListEqual(res0[0], [[1, 0], [0, 0], [0, 0]])
        self.assertListEqual(res0[1], [[0, 1], [0, 0], [0, 0]])
        self.assertListEqual(res0[2], [[0, 0], [1, 1], [0, 0]])
        self.assertListEqual(res0[3], [[0, 0], [0, 0], [1, 0]])
        self.assertListEqual(res0[4], [[0, 0], [0, 0], [0, 1]])

    def testHottifySynthetic1D(self):
        mask = np.array([0, 1, 2, 2, 3, 4])
        res = onehottify(mask)
        self.assertEqual(res.shape, (6, 5))
        res0 = np.transpose(res, (1, 0)).tolist()
        self.assertListEqual(res0[0], [1, 0, 0, 0, 0, 0])
        self.assertListEqual(res0[1], [0, 1, 0, 0, 0, 0])
        self.assertListEqual(res0[2], [0, 0, 1, 1, 0, 0])
        self.assertListEqual(res0[3], [0, 0, 0, 0, 1, 0])
        self.assertListEqual(res0[4], [0, 0, 0, 0, 0, 1])

    def testColapseSynthetic2D(self):
        mask = np.array([[[1, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0]],
                         [[0, 0, 1, 0, 0],
                          [0, 0, 1, 0, 0]],
                         [[0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 1]]])
        res = cp.labelPlanarInstances(mask, sort=False)
        res = res.tolist()
        self.assertListEqual(res, [[[0], [1]], [[2], [2]], [[3], [4]]])

    def testCollapse(self):
        raw_planes = []
        planes = []
        onehots = []
        filenames = sorted(glob('./test/**/*_planes_*.tiff', recursive=True))
        print(filenames)
        for filename in filenames:
            raw = read_tiff(filename)
            raw[:, :, 0] = 1 - np.sign(np.sum(raw[:, :, 1:], axis=2))
            print(raw.shape)
            raw_planes.append(raw)

            planes.append(cp.labelPlanarInstances(raw_planes[-1]))
            onehots.append(onehottify(planes[-1]))
            print(raw_planes[-1].shape, planes[-1].shape, onehots[-1].shape)

        for raw_p, reconst_p, colapsed_p in zip(raw_planes, onehots, planes):
            self.assertTupleEqual(raw_p.shape, reconst_p.shape, "Same shape")
            self.assertEqual(np.min(colapsed_p), 0, "Minimal plane label")
            self.assertEqual(np.max(colapsed_p), raw_p.shape[2] - 1, "Maximal plane label")
            for i in range(raw_p.shape[2]):
                self.assertEqual(raw_p[:, :, i].sum(), reconst_p[:, :, i].sum(),
                                 "Same count of ones " + str(i))
                self.assertEqual(np.sum(np.abs(raw_p[:, :, i] - reconst_p[:, :, i])), 0,
                                 "Same positions of ones " + str(i))

            non_planar_orig = raw_p[:, :, 0]
            non_planar_label = np.squeeze(1 - np.abs(np.sign(colapsed_p)))
            non_planar_hot = 1 - np.sign(np.sum(reconst_p[:, :, 1:], axis=2))
            print(non_planar_hot.sum(), "aaaaa", non_planar_label.sum())
            self.assertEqual(np.sum(np.abs(non_planar_orig - non_planar_hot)), 0,
                             "Matching non planar regions 1")
            self.assertEqual(np.sum(np.abs(non_planar_orig - non_planar_label)), 0,
                             "Matching non planar regions 2")


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestColapsePlanes)
    unittest.TextTestRunner(verbosity=2).run(suite)
