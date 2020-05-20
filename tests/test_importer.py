import random
from unittest import TestCase

from shapely.geometry import Polygon

from city_graph import importer


class TractableGridTesselationTestCase(TestCase):
    """
    This test case checks everything on a very small square grid --
    2x2.  For such a small grid every calculation should be tractable
    and we can check everything directly.
    """

    def setUp(self):
        """
        Take a polygon of 100x100 meters and split into a grid of
        50x50 meter cells.
        """
        polygon = Polygon([(0, 0), (0, 100), (100, 100), (100, 0)])
        self.tesselation = importer.GridTesselation(polygon, resolution=50)

    def test_grid_tesselation_constructed_properly(self):
        """
        Examine the tesselation and test that all the tesselation
        parameters are calculated correctly.
        """
        self.assertEqual(self.tesselation.xmin, 0)
        self.assertEqual(self.tesselation.ymin, 0)
        self.assertEqual(self.tesselation.xmax, 100)
        self.assertEqual(self.tesselation.ymax, 100)
        self.assertEqual(self.tesselation.nx, 2)
        self.assertEqual(self.tesselation.ny, 2)
        self.assertEqual(self.tesselation.ncells, 4)

    def test_grid_projects_cell_center_into_an_expected_cell_and_into_itself(self):
        """
        Test that projecting a cell center lands it into the
        corresponding cell.
        """

        index = 0
        for iy in range(0, 2):
            for ix in range(0, 2):
                x0 = ix * 50 + 25
                y0 = iy * 50 + 25

                self.assertEqual(self.tesselation._ixiy(x0, y0), (ix, iy))
                self.assertEqual(self.tesselation.index(x0, y0), index)

                x1, y1 = self.tesselation.center(index)
                self.assertAlmostEqual(x0, x1)
                self.assertAlmostEqual(y0, y1)

                index += 1

    def test_corners_stay_inside_the_grid(self):
        """
        Project the corners and check that they stay inside the
        region.
        """
        in_out_points = [
            ((  0,   0), (25, 25)),
            ((  0, 100), (25, 75)),
            ((100,   0), (75, 25)),
            ((100, 100), (75, 75))
        ]

        for in_, out in in_out_points:
            x0, y0 = in_
            x1_expected, y1_expected = out

            index = self.tesselation.index(x0, y0)
            x1_projected, y1_projected = self.tesselation.center(index)

            self.assertAlmostEqual(x1_expected, x1_projected)
            self.assertAlmostEqual(y1_expected, y1_projected)


class RandomGridTesselationTestCase(TestCase):
    def setUp(self):
        pass
