import sys
import unittest
import spacy


sys.path.insert(1,'..')
from bible_two_list_model import find_matches_from_both_lists,   combine_matches, create_feature_array_from_matches, parse_line_into_components, is_new_testament

class TestBibleTwoListModel(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.nlp = spacy.load('en_core_web_sm')

  def test_find_matches_from_both_lists_no_matches(self):
    test_results = find_matches_from_both_lists(self.nlp(''), [], [] )
    self.assertEqual(test_results['matches_from_both'], False, "should return false")
    self.assertEqual(len(test_results['matches_a']), 0, "should be empty")
    self.assertEqual(len(test_results['matches_b']), 0, "should be empty")

    test_results = find_matches_from_both_lists(self.nlp(''), ['one', 'two','three'], ['four','five'])
    self.assertEqual(test_results['matches_from_both'], False, "should return false")
    self.assertEqual(len(test_results['matches_a']), 0, "should be empty")
    self.assertEqual(len(test_results['matches_b']), 0, "should be empty")

    test_results = find_matches_from_both_lists(self.nlp('I am getting paid by the line of code'), ['one', 'two','three'], ['four','five'])
    self.assertEqual(test_results['matches_from_both'], False, "should return false")
    self.assertEqual(len(test_results['matches_a']), 0, "should be empty")
    self.assertEqual(len(test_results['matches_b']), 0, "should be empty")

  def test_find_matches_from_both_lists_with_matches(self):
    test_results = find_matches_from_both_lists(self.nlp('I am getting paid by the line of code'), ['pay'], 
       [  'code'])
    self.assertEqual(test_results['matches_from_both'], True, "should return true")
    self.assertEqual(len(test_results['matches_a']), 1, "should have one element")
    self.assertEqual(len(test_results['matches_b']), 1, "should have one element")

    test_results = find_matches_from_both_lists(self.nlp('I am getting paid by the line of code'), ['be','get', 'pay'], 
       [ 'line','of', 'code'])
    self.assertEqual(test_results['matches_from_both'], True, "should return true")
    self.assertEqual(len(test_results['matches_a']), 3, "should have three elements")
    self.assertEqual(len(test_results['matches_b']), 3, "should have three elements")

  def test_combine_matches(self):
    test_results = combine_matches([], [])
    self.assertEqual(len(test_results), 0, "should be empty")

    test_results = combine_matches(['a'], [])
    self.assertEqual(len(test_results), 0, "should be empty")

    test_results = combine_matches([], ['b'])
    self.assertEqual(len(test_results), 0, "should be empty")    

    test_results = combine_matches(['b','c','d'], ['a', 'e'])
    self.assertEqual(len(test_results), 6, "should be 6 elements")   
    self.assertTrue('ab' in test_results)
    self.assertTrue('be' in test_results)
    self.assertTrue('ac' in test_results)
    self.assertTrue('ce' in test_results)
    self.assertTrue('ad' in test_results)
    self.assertTrue('de' in test_results)

  def test_create_feature_array_from_matches(self):
    possible_matches = ['ab','be','ac','ce','ad','de']     # see above test

    test_results = create_feature_array_from_matches([], possible_matches)
    for tr in test_results:
      self.assertEqual(tr,0)
    
    test_results = create_feature_array_from_matches(['ab'], possible_matches)
    self.assertEqual(test_results[0], 1)
    for i in range(1,5):
      self.assertEqual(test_results[i],0)

    test_results = create_feature_array_from_matches(['ab','ac','de'], possible_matches)
    self.assertEqual(test_results[0], 1)
    self.assertEqual(test_results[1], 0)
    self.assertEqual(test_results[2], 1)
    self.assertEqual(test_results[3], 0)
    self.assertEqual(test_results[4], 0)
    self.assertEqual(test_results[5], 1)
     
  def test_parse_line_into_components(self):
    test_results = parse_line_into_components('Psa25:22 Redeem Israel, O God, out of all his troubles.')
    self.assertEquals(test_results['book'], 'Psa')
    self.assertEquals(test_results['chapter'], '25')
    self.assertEquals(test_results['verse'], '22')
    self.assertEquals(test_results['text'], 'Redeem Israel, O God, out of all his troubles.')

  def test_is_new_testament(self):
    self.assertFalse(is_new_testament('Gen'))
    self.assertTrue(is_new_testament('Mat'))

if __name__ ==  '__main__':
  unittest.main()