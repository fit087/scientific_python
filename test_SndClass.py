import unittest as ut
import SndClass as sec

class test2ndClass(ut.TestCase):

	def test_farentheit(self):
		self.assertEqual(sec.farentheit(0), 32)
		self.assertEqual(sec.farentheit(2.5), 36.5)
		#self.assertAlmostEqual(sec.farentheit(37.7778), 100)
		self.assertTrue(self.residual(sec.farentheit(37.7778), 100) < 1e-3)
		self.assertTrue((sec.farentheit(37.7778) - 100) < 1e-3)
		self.assertEqual(sec.farentheit(100), 212)

	@staticmethod
	def residual(a, b):
		return a - b

if __name__ == '__main__':
	ut.main()
