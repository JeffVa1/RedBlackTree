import unittest
import types
from Project3.RBtree import RBtree
from Project3.RBnode import RBnode
from random import randint, seed

seed(331)


class TestProject3(unittest.TestCase):

    def test_static_methods(self):
        """
        Tests for set_child, replace_child, get_sibling, get_grandparent, and get_uncle
        """
        # build testing tree
        # we call static methods on tree1, and hardcode changes to tree2
        # and test equality to check whether methods function correctly
        root = RBnode(5, False)
        root.left = RBnode(3, True, root); root.right = RBnode(7, True, root)
        root.left.left = RBnode(2, False, root.left)
        root.left.right = RBnode(4, False, root.left)
        root.right.left = RBnode(6, False, root.right)
        root.right.right = RBnode(8, False, root.right)
        root.left.left.right = RBnode(2.5, True, root.left.left)
        tree = RBtree(root)
        tree2 = RBtree(root)

        RBtree.set_child(tree.root.right.right, RBnode(9, True), False)
        tree2.root.right.right.right = RBnode(9, True, tree2.root.right.right)
        assert tree == tree2

        RBtree.set_child(tree.root.right.right, RBnode(7.5, True), True)
        tree2.root.right.right.left = RBnode(7.5, True, tree2.root.right)
        assert tree == tree2

        RBtree.replace_child(tree.root.left.left, tree.root.left.left.right, RBnode(2.4, True))
        tree2.root.left.left.right = RBnode(2.4, True, tree2.root.left.right)
        assert tree == tree2

        RBtree.replace_child(tree.root.right.right, tree.root.right.right.right, RBnode(3.1, True))
        tree2.root.right.right.right = RBnode(3.1, True, tree2.root.right.right)
        assert tree == tree2

        self.assertEqual(None, RBtree.get_sibling(tree.root))
        self.assertEqual(None, RBtree.get_uncle(tree.root))
        self.assertEqual(None, RBtree.get_grandparent(tree.root))

        self.assertEqual(tree.root.right, RBtree.get_sibling(tree.root.left))
        self.assertEqual(None, RBtree.get_uncle(tree.root.left))
        self.assertEqual(None, RBtree.get_grandparent(tree.root.left))

        self.assertEqual(tree.root.left, RBtree.get_sibling(tree.root.right))
        self.assertEqual(None, RBtree.get_uncle(tree.root.right))
        self.assertEqual(None, RBtree.get_grandparent(tree.root.right))

        self.assertEqual(tree.root.left, RBtree.get_uncle(tree.root.right.right))
        self.assertEqual(tree.root, RBtree.get_grandparent(tree.root.right.right))

        self.assertEqual(tree.root.right, RBtree.get_uncle(tree.root.left.left))
        self.assertEqual(tree.root, RBtree.get_grandparent(tree.root.left.left))

    def test_rotations_simple(self):
        """ all nodes colored black to maintain RB property """

        # Simple tree, rotate left at root
        tree = RBtree()
        tree.root = RBnode(1, False)
        tree.root.right = RBnode(2, False, parent=tree.root)
        tree.root.right.right = RBnode(3, False, parent=tree.root.right)
        """
        Initial Structure:
          1 (B) (None)
                       2 (B) (1)
                                 3 (B) (2)
        Expected Final Structure:
                                2 (B) (None)
                      1 (B) (2)              3 (B) (2)
        """
        tree.left_rotate(tree.root)

        self.assertEqual(2, tree.root.value)
        self.assertEqual(None, tree.root.parent)
        self.assertEqual(1, tree.root.left.value)
        self.assertEqual(None, tree.root.left.left)
        self.assertEqual(None, tree.root.left.right)
        self.assertEqual(tree.root, tree.root.left.parent)
        self.assertEqual(3, tree.root.right.value)
        self.assertEqual(None, tree.root.right.left)
        self.assertEqual(None, tree.root.right.right)
        self.assertEqual(tree.root, tree.root.right.parent)

        # left rotation on more complex tree
        tree = RBtree()
        tree.root = RBnode(4, False)
        tree.root.left = RBnode(2, False, parent=tree.root)
        tree.root.right = RBnode(6, False, parent=tree.root)
        tree.root.right.right = RBnode(8, False, parent=tree.root.right)
        tree.root.right.right.right = RBnode(10, False, parent=tree.root.right.right)
        """
        Initial Structure:
                            4 (B) (None)
                 2 (B) (4)                6 (B) (4)
                                                     8 (B) (6)
                                                                 10 (B) (8)
        Expected Final Structure:
                           4 (B) (None)
               2 (B) (4)                    8 (B) (4)
                                6 (B) (8)               10 (B) (8)
        """
        tree.left_rotate(tree.root.right)

        self.assertEqual(4, tree.root.value)
        self.assertEqual(None, tree.root.parent)
        self.assertEqual(2, tree.root.left.value)
        self.assertEqual(tree.root, tree.root.left.parent)
        self.assertEqual(None, tree.root.left.left)
        self.assertEqual(None, tree.root.left.right)
        self.assertEqual(8, tree.root.right.value)
        self.assertEqual(tree.root, tree.root.right.parent)
        self.assertEqual(6, tree.root.right.left.value)
        self.assertEqual(tree.root.right, tree.root.right.left.parent)
        self.assertEqual(None, tree.root.right.left.left)
        self.assertEqual(None, tree.root.right.left.right)
        self.assertEqual(10, tree.root.right.right.value)
        self.assertEqual(tree.root.right, tree.root.right.right.parent)
        self.assertEqual(None, tree.root.right.right.left)
        self.assertEqual(None, tree.root.right.right.right)

        # simple tree, rotate right at root
        tree = RBtree()
        tree.root = RBnode(3, False)
        tree.root.left = RBnode(2, False, parent=tree.root)
        tree.root.left.left = RBnode(1, False, parent=tree.root.left)
        """
        Initial Structure:
                            3 (B) (None)
                   2 (B) (3)
            1 (B) (2)
        Expected Final Structure:
                                    2 (B) (None)
                        1 (B) (2)                 3 (B) (2)
        """
        tree.right_rotate(tree.root)

        self.assertEqual(2, tree.root.value)
        self.assertEqual(None, tree.root.parent)
        self.assertEqual(1, tree.root.left.value)
        self.assertEqual(None, tree.root.left.left)
        self.assertEqual(None, tree.root.left.right)
        self.assertEqual(tree.root, tree.root.left.parent)
        self.assertEqual(3, tree.root.right.value)
        self.assertEqual(None, tree.root.right.left)
        self.assertEqual(None, tree.root.left.right)
        self.assertEqual(tree.root, tree.root.right.parent)

        # right rotation on more complex tree
        tree = RBtree()
        tree.root = RBnode(7, False)
        tree.root.left = RBnode(3, False, parent=tree.root)
        tree.root.left.left = RBnode(2, False, parent=tree.root.left)
        tree.root.left.left.left = RBnode(1, False, parent=tree.root.left.left)
        tree.root.left.right = RBnode(4, False, parent=tree.root.left)
        tree.root.right = RBnode(10, False, parent=tree.root)
        """
        Initial Structure:
                                                7 (B) (None)
                                    3 (B) (7)                   10 (B) (7)
                          2 (B) (3)           4 (B) (3)
             1 (B) (2)
        Expected Final Structure:
                                     3 (B) (None)
                            2 (B) (3)             7 (B) (3)
                    1 (B) (3)            4 (B) (7)          10 (B) (7)
        """
        tree.right_rotate(tree.root)

        self.assertEqual(3, tree.root.value)
        self.assertEqual(None, tree.root.parent)
        self.assertEqual(2, tree.root.left.value)
        self.assertEqual(tree.root, tree.root.left.parent)
        self.assertEqual(1, tree.root.left.left.value)
        self.assertEqual(tree.root.left, tree.root.left.left.parent)
        self.assertEqual(None, tree.root.left.left.left)
        self.assertEqual(None, tree.root.left.left.right)
        self.assertEqual(7, tree.root.right.value)
        self.assertEqual(tree.root, tree.root.right.parent)
        self.assertEqual(4, tree.root.right.left.value)
        self.assertEqual(tree.root.right, tree.root.right.left.parent)
        self.assertEqual(None, tree.root.right.left.left)
        self.assertEqual(None, tree.root.right.left.right)
        self.assertEqual(10, tree.root.right.right.value)
        self.assertEqual(tree.root.right, tree.root.right.right.parent)
        self.assertEqual(None, tree.root.right.right.left)
        self.assertEqual(None, tree.root.right.right.right)

    def test_rotations_complex(self):
        """ all nodes colored black to maintain Red/Black property """

        # Test 1

        tree = RBtree()
        tree.root = RBnode(7, False)
        tree.root.left = RBnode(3, False, parent=tree.root)
        tree.root.right = RBnode(10, False, parent=tree.root)
        tree.root.right.left = RBnode(9, False, parent=tree.root.right)
        tree.root.right.right = RBnode(11, False, parent=tree.root.right)
        tree.root.right.right.right = RBnode(12, False, parent=tree.root.right.right)
        """
        Initial Structure:
                                		7
                                	   / \
                                	  3  10
                                	    /  \
                                	   9   11
                                	        \
                                	         12
        Expected Final Structure:
                                		10
                                	   /  \
                                	  7   11
                                	 / \    \
                                	3   9    12
        """
        tree.left_rotate(tree.root)

        self.assertEqual(10, tree.root.value)
        self.assertEqual(None, tree.root.parent)
        self.assertEqual(7, tree.root.left.value)
        self.assertEqual(tree.root, tree.root.left.parent)
        self.assertEqual(3, tree.root.left.left.value)
        self.assertEqual(tree.root.left, tree.root.left.left.parent)
        self.assertEqual(None, tree.root.left.left.left)
        self.assertEqual(None, tree.root.left.left.right)
        self.assertEqual(9, tree.root.left.right.value)
        self.assertEqual(tree.root.left, tree.root.left.right.parent)
        self.assertEqual(None, tree.root.left.right.left)
        self.assertEqual(None, tree.root.left.right.right)
        self.assertEqual(11, tree.root.right.value)
        self.assertEqual(tree.root, tree.root.right.parent)
        self.assertEqual(None, tree.root.right.left)
        self.assertEqual(12, tree.root.right.right.value)
        self.assertEqual(tree.root.right, tree.root.right.right.parent)
        self.assertEqual(None, tree.root.right.right.left)
        self.assertEqual(None, tree.root.right.right.right)

        # Test 2

        tree = RBtree()
        tree.root = RBnode(10, False)
        tree.root.right = RBnode(11, False, parent=tree.root)
        tree.root.right.right = RBnode(12, False, parent=tree.root.right)
        tree.root.left = RBnode(5, False, parent=tree.root)
        tree.root.left.right = RBnode(7, False, parent=tree.root.left)
        tree.root.left.left = RBnode(3, False, parent=tree.root.left)
        tree.root.left.left.right = RBnode(4, False, parent=tree.root.left.left)
        tree.root.left.left.left = RBnode(2, False, parent=tree.root.left.left)
        tree.root.left.left.left.left = RBnode(1, False, parent=tree.root.left.left.left)
        """
        Initial Structure:
                        			10
                        		   /  \
                        		  5	   11
                        		 / \     \
                        		3	7    12
                        	   / \
                        	  2   4
                             /
                            1
        Expected Final Structure:
                            		10
                            	   /  \
                            	  3    11
                            	 / \     \
                            	2   5     12
                               /   / \
                              1   4   7
        """
        tree.right_rotate(tree.root.left)

        self.assertEqual(10, tree.root.value)
        self.assertEqual(None, tree.root.parent)
        self.assertEqual(11, tree.root.right.value)
        self.assertEqual(tree.root, tree.root.right.parent)
        self.assertEqual(None, tree.root.right.left)
        self.assertEqual(12, tree.root.right.right.value)
        self.assertEqual(tree.root.right, tree.root.right.right.parent)
        self.assertEqual(None, tree.root.right.right.left)
        self.assertEqual(None, tree.root.right.right.right)
        self.assertEqual(5, tree.root.left.right.value)
        self.assertEqual(tree.root, tree.root.left.parent)
        self.assertEqual(5, tree.root.left.right.value)
        self.assertEqual(tree.root.left, tree.root.left.right.parent)
        self.assertEqual(4, tree.root.left.right.left.value)
        self.assertEqual(tree.root.left.right, tree.root.left.right.left.parent)
        self.assertEqual(None, tree.root.left.right.left.left)
        self.assertEqual(None, tree.root.left.right.left.right)
        self.assertEqual(7, tree.root.left.right.right.value)
        self.assertEqual(tree.root.left.right, tree.root.left.right.right.parent)
        self.assertEqual(None, tree.root.left.right.right.left)
        self.assertEqual(None, tree.root.left.right.right.right)
        self.assertEqual(2, tree.root.left.left.value)
        self.assertEqual(tree.root.left, tree.root.left.left.parent)
        self.assertEqual(None, tree.root.left.left.right)
        self.assertEqual(1, tree.root.left.left.left.value)
        self.assertEqual(tree.root.left.left, tree.root.left.left.left.parent)
        self.assertEqual(None, tree.root.left.left.left.left)
        self.assertEqual(None, tree.root.left.left.left.right)

        # Test 3

        tree = RBtree()
        tree.root = RBnode(3, False)
        tree.root.left = RBnode(2, False, parent=tree.root)
        tree.root.left.left = RBnode(1, False, parent=tree.root.left)
        tree.root.right= RBnode(10, False, parent=tree.root)
        tree.root.right.left = RBnode(5, False, parent=tree.root.right)
        tree.root.right.right = RBnode(12, False, parent=tree.root.right)
        tree.root.right.right.left = RBnode(11, False, parent=tree.root.right)
        tree.root.right.right.right = RBnode(13, False, parent=tree.root.right.right)
        tree.root.right.right.right.right = RBnode(14, False, parent=tree.root.right.right.right)
        """
        Initial Structure:		3
                        	   / \
                        	  2   10
                        	 /   /  \
                        	1   5   12
                        	       /  \
                        	      11   13
                        	             \
                        	              14
        Expected Final Structure:
                        		3
                        	   / \
                        	  2   12
                        	 /   /  \
                        	1   10   13
                        	   /  \    \
                        	  5   11   14
        """
        tree.left_rotate(tree.root.right)

        self.assertEqual(3, tree.root.value)
        self.assertEqual(None, tree.root.parent)
        self.assertEqual(2, tree.root.left.value)
        self.assertEqual(tree.root, tree.root.left.parent)
        self.assertEqual(None, tree.root.left.right)
        self.assertEqual(1, tree.root.left.left.value)
        self.assertEqual(tree.root.left, tree.root.left.left.parent)
        self.assertEqual(None, tree.root.left.left.left)
        self.assertEqual(None, tree.root.left.left.right)
        self.assertEqual(12, tree.root.right.value)
        self.assertEqual(tree.root, tree.root.right.parent)
        self.assertEqual(10, tree.root.right.left.value)
        self.assertEqual(tree.root.right, tree.root.right.left.parent)
        self.assertEqual(5, tree.root.right.left.left.value)
        self.assertEqual(tree.root.right.left, tree.root.right.left.left.parent)
        self.assertEqual(None, tree.root.right.left.left.left)
        self.assertEqual(None, tree.root.right.left.left.right)
        self.assertEqual(11, tree.root.right.left.right.value)
        self.assertEqual(tree.root.right.left, tree.root.right.left.right.parent)
        self.assertEqual(None, tree.root.right.left.right.left)
        self.assertEqual(None, tree.root.right.left.right.right)
        self.assertEqual(13, tree.root.right.right.value)
        self.assertEqual(tree.root.right, tree.root.right.right.parent)
        self.assertEqual(None, tree.root.right.right.left)
        self.assertEqual(14, tree.root.right.right.right.value)
        self.assertEqual(tree.root.right.right, tree.root.right.right.right.parent)
        self.assertEqual(None, tree.root.right.right.right.left)
        self.assertEqual(None, tree.root.right.right.right.right)

    def test_traversals(self):

        # base case - empty
        tree = RBtree()
        gen1 = tree.preorder(tree.root)
        gen2 = tree.postorder(tree.root)
        gen3 = tree.inorder(tree.root)
        gen4 = tree.bfs(tree.root)

        assert isinstance(gen1, types.GeneratorType) and next(gen1, None) is None
        assert isinstance(gen2, types.GeneratorType) and next(gen2, None) is None
        assert isinstance(gen3, types.GeneratorType) and next(gen3, None) is None
        assert isinstance(gen4, types.GeneratorType) and next(gen4, None) is None

        # base case - one element
        tree = RBtree()
        tree.root = RBnode(14, False)
        gen5 = tree.preorder(tree.root)
        gen6 = tree.postorder(tree.root)
        gen7 = tree.inorder(tree.root)
        gen8 = tree.bfs(tree.root)

        assert next(gen5, None).value == 14
        assert next(gen5, None) is None
        assert next(gen6, None).value == 14
        assert next(gen6, None) is None
        assert next(gen7, None).value == 14
        assert next(gen7, None) is None
        assert next(gen8, None).value == 14
        assert next(gen8, None) is None

        # small tree
        tree = RBtree()
        tree.root = RBnode(14, False)
        tree.root.left = RBnode(7, False, parent=tree.root)
        tree.root.left.left = RBnode(3, True, parent=tree.root.left)
        tree.root.left.right = RBnode(10, True, parent=tree.root.left)
        tree.root.right = RBnode(21, False, parent=tree.root)
        tree.root.right.left = RBnode(17, True, parent=tree.root.right)
        tree.root.right.right = RBnode(25, True, parent=tree.root.right)

        gen9 = tree.preorder(tree.root)
        gen10 = tree.postorder(tree.root)
        gen11 = tree.inorder(tree.root)
        gen12 = tree.bfs(tree.root)

        pre = [14, 7, 3, 10, 21, 17, 25]
        post = [3, 10, 7, 17, 25, 21, 14]
        inorder = [3, 7, 10, 14, 17, 21, 25]
        bfs = [14, 7, 21, 3, 10, 17, 25]
        print(next(gen10))
        print(next(gen10))
        print(next(gen10))
        print(next(gen10))
        print(next(gen10))
        print(next(gen10))
        print(next(gen10))
        for i in range(7):
            assert next(gen9, None).value == pre[i]  # 1
            assert next(gen10, None).value == post[i]  # 2
            assert next(gen11, None).value == inorder[i]  # 3
            assert next(gen12, None).value == bfs[i]  # 4

    def test_max_min(self):
        # empty
        tree = RBtree()
        assert tree.min(tree.root) is None
        assert tree.max(tree.root) is None

        # one element
        tree.root = RBnode(15, False)
        assert tree.min(tree.root).value == 15
        assert tree.max(tree.root).value == 15

        # small tree
        tree.root.left = RBnode(4, False, parent=tree.root)
        tree.root.left.left = RBnode(2, True, parent=tree.root.left)
        tree.root.left.right = RBnode(10, True, parent=tree.root.left)
        tree.root.right = RBnode(22, False, parent=tree.root)
        tree.root.right.left = RBnode(16, True, parent=tree.root.right)
        tree.root.right.right = RBnode(37, True, parent=tree.root.right)

        assert tree.min(tree.root).value == 2
        assert tree.max(tree.root).value == 37
        assert tree.min(tree.root) == RBnode(2, True, parent=tree.root.left)
        assert tree.max(tree.root) == RBnode(37, True, parent=tree.root.right)

    def test_search(self):
        # empty
        tree = RBtree()
        self.assertEqual(tree.search(tree.root, 1), None)

        # small
        tree.root = RBnode(20, False)
        tree.root.left = RBnode(10, False, parent=tree.root)
        tree.root.left.left = RBnode(5, True, parent=tree.root.left)
        tree.root.left.right = RBnode(15, True, parent=tree.root.left)
        tree.root.right = RBnode(30, False, parent=tree.root)
        tree.root.right.left = RBnode(25, True, parent=tree.root.right)
        tree.root.right.right = RBnode(35, True, parent=tree.root.right)

        self.assertEqual(tree.search(tree.root, 20), tree.root)
        self.assertEqual(tree.search(tree.root, 5), tree.root.left.left)
        self.assertEqual(tree.search(tree.root, 35), tree.root.right.right)
        self.assertEqual(tree.search(tree.root, 10), tree.root.left)
        self.assertEqual(tree.search(tree.root, 15), tree.root.left.right)
        self.assertEqual(tree.search(tree.root, 25), tree.root.right.left)
        self.assertEqual(tree.search(tree.root, 30), tree.root.right)

        # a bit more complex
        tree.root.left = RBnode(10, True, parent=tree.root)
        tree.root.left.left = RBnode(5, False, parent=tree.root.left)
        tree.root.left.right = RBnode(15, False, parent=tree.root.left)
        tree.root.right = RBnode(30, True, parent=tree.root)
        tree.root.right.left = RBnode(25, False, parent=tree.root.right)
        tree.root.right.right = RBnode(35, False, parent=tree.root.right)
        tree.root.left.left.left = RBnode(3, True, parent=tree.root.left)
        tree.root.left.right.right = RBnode(17, True, parent=tree.root.left)
        tree.root.right.right.left = RBnode(32, True, parent=tree.root.left)

        self.assertEqual(tree.search(tree.root, 3), tree.root.left.left.left)
        self.assertEqual(tree.search(tree.root, 2), tree.root.left.left.left)  # potential parent
        self.assertEqual(tree.search(tree.root, 4), tree.root.left.left.left)  # potential parent
        self.assertEqual(tree.search(tree.root, 17), tree.root.left.right.right)
        self.assertEqual(tree.search(tree.root, 27), tree.root.right.left )  # potential parent

    def test_insert_basic(self):
        # insert root
        tree = RBtree()
        tree.insert(tree.root, 12)

        self.assertEqual(tree.root.value, 12)
        self.assertEqual(tree.root.is_red, False)
        self.assertEqual(tree.root.left, None)
        self.assertEqual(tree.root.right, None)
        self.assertEqual(tree.root.parent, None)

        # parent is black
        tree.insert(tree.root, 4)
        tree.insert(tree.root, 16)

        self.assertEqual(tree.root.value, 12)
        self.assertEqual(tree.root.is_red, False)
        self.assertEqual(tree.root.left.value, 4)
        self.assertEqual(tree.root.left.is_red, True)
        self.assertEqual(tree.root.left.parent.left.value, 4)
        self.assertEqual(tree.root.right.value, 16)
        self.assertEqual(tree.root.right.is_red, True)
        self.assertEqual(tree.root.right.parent.right.value, 16)

        # single left rotation
        tree2 = RBtree()
        tree2.insert(tree2.root, 4)
        tree2.insert(tree2.root, 12)
        tree2.insert(tree2.root, 17)

        self.assertEqual(tree2.root.value, 12)
        self.assertEqual(tree2.root.is_red, False)
        self.assertEqual(tree2.root.parent, None)
        self.assertEqual(tree2.root.left.value, 4)
        self.assertEqual(tree2.root.left.is_red, True)
        self.assertEqual(tree2.root.left.parent.left.value, 4)
        self.assertEqual(tree2.root.left.left, None)
        self.assertEqual(tree2.root.left.right, None)
        self.assertEqual(tree2.root.right.value, 17)
        self.assertEqual(tree2.root.right.is_red, True)
        self.assertEqual(tree2.root.right.parent.right.value, 17)
        self.assertEqual(tree2.root.right.left, None)
        self.assertEqual(tree2.root.right.right, None)


        # single right rotation
        tree3 = RBtree()
        tree3.insert(tree3.root, 24)
        tree3.insert(tree3.root, 12)
        tree3.insert(tree3.root, 6)

        self.assertEqual(tree3.root.value, 12)
        self.assertEqual(tree3.root.is_red, False)
        self.assertEqual(tree3.root.parent, None)
        self.assertEqual(tree3.root.left.value, 6)
        self.assertEqual(tree3.root.left.is_red, True)
        self.assertEqual(tree3.root.left.parent.left.value, 6)
        self.assertEqual(tree3.root.left.left, None)
        self.assertEqual(tree3.root.left.right, None)
        self.assertEqual(tree3.root.right.value, 24)
        self.assertEqual(tree3.root.right.is_red, True)
        self.assertEqual(tree3.root.right.parent.right.value, 24)
        self.assertEqual(tree3.root.right.left, None)
        self.assertEqual(tree3.root.right.right, None)

        # parent and uncle red case
        tree4 = RBtree()
        tree4.insert(tree4.root, 24)
        tree4.insert(tree4.root, 12)
        tree4.insert(tree4.root, 6)
        tree4.insert(tree4.root, 9)

        assert tree4.root.value == 12
        assert not tree4.root.is_red
        assert tree4.root.left.value == 6
        assert not tree4.root.left.is_red
        assert tree4.root.right.value == 24
        assert not tree4.root.right.is_red
        assert tree4.root.left.right.value == 9
        assert tree4.root.left.right.is_red
        self.assertEqual(tree4.root.value, 12)
        self.assertEqual(tree4.root.is_red, False)
        self.assertEqual(tree4.root.parent, None)
        self.assertEqual(tree4.root.left.value, 6)
        self.assertEqual(tree4.root.left.is_red, False)
        self.assertEqual(tree4.root.left.parent.left.value, 6)
        self.assertEqual(tree4.root.left.left, None)
        self.assertEqual(tree4.root.left.right.value, 9)
        self.assertEqual(tree4.root.left.right.is_red, True)
        self.assertEqual(tree4.root.left.right.parent.right.value, 9)
        self.assertEqual(tree4.root.left.right.left, None)
        self.assertEqual(tree4.root.left.right.right, None)
        self.assertEqual(tree4.root.right.value, 24)
        self.assertEqual(tree4.root.right.is_red, False)
        self.assertEqual(tree4.root.right.parent.right.value, 24)
        self.assertEqual(tree4.root.right.left, None)
        self.assertEqual(tree4.root.right.right, None)

        # node is parent's left child and parent is grandparent's right child
        tree5 = RBtree()
        tree5.insert(tree5.root, 12)
        tree5.insert(tree5.root, 26)
        tree5.insert(tree5.root, 18)

        self.assertEqual(tree5.root.value, 18)
        self.assertEqual(tree5.root.is_red, False)
        self.assertEqual(tree5.root.parent, None)
        self.assertEqual(tree5.root.left.value, 12)
        self.assertEqual(tree5.root.left.is_red, True)
        self.assertEqual(tree5.root.left.parent.left.value, 12)
        self.assertEqual(tree5.root.left.left, None)
        self.assertEqual(tree5.root.left.right, None)
        self.assertEqual(tree5.root.right.value, 26)
        self.assertEqual(tree5.root.right.is_red, True)
        self.assertEqual(tree5.root.right.parent.right.value, 26)
        self.assertEqual(tree5.root.right.left, None)
        self.assertEqual(tree5.root.right.right, None)

        # node is parent's right child and parent is grandparent's left child
        tree6 = RBtree()
        tree6.insert(tree6.root, 25)
        tree6.insert(tree6.root, 13)
        tree6.insert(tree6.root, 19)

        self.assertEqual(tree6.root.value, 19)
        self.assertEqual(tree6.root.is_red, False)
        self.assertEqual(tree6.root.parent, None)
        self.assertEqual(tree6.root.left.value, 13)
        self.assertEqual(tree6.root.left.is_red, True)
        self.assertEqual(tree6.root.left.parent.left.value, 13)
        self.assertEqual(tree6.root.left.left, None)
        self.assertEqual(tree6.root.left.right, None)
        self.assertEqual(tree6.root.right.value, 25)
        self.assertEqual(tree6.root.right.is_red, True)
        self.assertEqual(tree6.root.right.parent.right.value, 25)
        self.assertEqual(tree6.root.right.left, None)
        self.assertEqual(tree6.root.right.right, None)

        # insert nodes already in a tree
        tree7 = RBtree()
        tree7.insert(tree7.root, 60)
        tree7.insert(tree7.root, 30)
        tree7.insert(tree7.root, 90)
        tree7.insert(tree7.root, 60)
        tree7.insert(tree7.root, 30)
        tree7.insert(tree7.root, 90)
        tree7.insert(tree7.root, 60)
        tree7.insert(tree7.root, 30)
        tree7.insert(tree7.root, 90)

        self.assertEqual(tree7.root.value, 60)
        self.assertEqual(tree7.root.is_red, False)
        self.assertEqual(tree7.root.parent, None)
        self.assertEqual(tree7.root.left.value, 30)
        self.assertEqual(tree7.root.left.is_red, True)
        self.assertEqual(tree7.root.left.parent.left.value, 30)
        self.assertEqual(tree7.root.left.left, None)
        self.assertEqual(tree7.root.left.right, None)
        self.assertEqual(tree7.root.right.value, 90)
        self.assertEqual(tree7.root.right.is_red, True)
        self.assertEqual(tree7.root.right.parent.right.value, 90)
        self.assertEqual(tree7.root.right.left, None)
        self.assertEqual(tree7.root.right.right, None)
        self.assertEqual(tree7.size, 3)

    def test_insert_mix(self):
        tree = RBtree()
        tree.insert(tree.root, 4)
        tree.insert(tree.root, 8)
        tree.insert(tree.root, 6)
        tree.insert(tree.root, 12)
        tree.insert(tree.root, 24)
        tree.insert(tree.root, 24)

        self.assertEqual(tree.root.value, 6)
        self.assertEqual(tree.root.is_red, False)
        self.assertEqual(tree.root.parent, None)
        self.assertEqual(tree.root.left.is_red, False)
        self.assertEqual(tree.root.right.left.is_red, True)
        self.assertEqual(tree.root.right.right.is_red, True)

        tree.insert(tree.root, 13)

        self.assertEqual(tree.root.right.is_red, True)
        self.assertEqual(tree.root.right.left.is_red, False)
        self.assertEqual(tree.root.right.right.left.is_red, True)
        self.assertEqual(tree.root.right.right.left.value, 13)

        tree.insert(tree.root, 100)
        tree.insert(tree.root, 1)

        self.assertEqual(tree.root.left.left.value, 1)
        self.assertEqual(tree.root.left.left.is_red, True)

        tree.insert(tree.root, 23)
        tree.insert(tree.root, 100)
        tree.insert(tree.root, 1)
        tree.insert(tree.root, 4)
        tree.insert(tree.root, 24)
        tree.insert(tree.root, 8)
        tree.insert(tree.root, 6)
        tree.insert(tree.root, 12)
        tree.insert(tree.root, 24)

        self.assertEqual(tree.root.value, 12)
        self.assertEqual(tree.root.is_red, False)
        self.assertEqual(tree.root.parent, None)
        self.assertEqual(tree.root.left.value, 6)
        self.assertEqual(tree.root.left.is_red, True)
        self.assertEqual(tree.root.left.parent.left.value, 6)
        self.assertEqual(tree.root.left.left.value, 4)
        self.assertEqual(tree.root.left.left.is_red, False)
        self.assertEqual(tree.root.left.left.parent.left.value, 4)
        self.assertEqual(tree.root.left.left.left.value, 1)
        self.assertEqual(tree.root.left.left.left.is_red, True)
        self.assertEqual(tree.root.left.left.left.parent.left.value, 1)
        self.assertEqual(tree.root.left.left.left.left, None)
        self.assertEqual(tree.root.left.left.left.right, None)
        self.assertEqual(tree.root.left.left.right, None)
        self.assertEqual(tree.root.left.right.value, 8)
        self.assertEqual(tree.root.left.right.is_red, False)
        self.assertEqual(tree.root.left.right.parent.right.value, 8)
        self.assertEqual(tree.root.left.right.left, None)
        self.assertEqual(tree.root.left.right.right, None)
        self.assertEqual(tree.root.right.value, 24)
        self.assertEqual(tree.root.right.is_red, True)
        self.assertEqual(tree.root.right.parent.right.value, 24)
        self.assertEqual(tree.root.right.left.value, 13)
        self.assertEqual(tree.root.right.left.is_red, False)
        self.assertEqual(tree.root.right.left.parent.left.value, 13)
        self.assertEqual(tree.root.right.left.left, None)
        self.assertEqual(tree.root.right.left.right.value, 23)
        self.assertEqual(tree.root.right.left.right.is_red, True)
        self.assertEqual(tree.root.right.left.right.parent.right.value, 23)
        self.assertEqual(tree.root.right.left.right.left, None)
        self.assertEqual(tree.root.right.left.right.right, None)
        self.assertEqual(tree.root.right.right.value, 100)
        self.assertEqual(tree.root.right.right.is_red, False)
        self.assertEqual(tree.root.right.right.parent.right.value, 100)
        self.assertEqual(tree.root.right.right.left, None)
        self.assertEqual(tree.root.right.right.right, None)

    def test_remove_basic(self):
        # remove empty tree
        tree = RBtree()
        tree.remove(tree.root, 2)

        # remove root node
        tree = RBtree()
        tree.root = RBnode(2, False)
        tree.remove(tree.root, 2)

        self.assertEqual(tree.root, None)

        # remove left child
        tree2 = RBtree()
        tree2.root = RBnode(2, False)
        tree2.root.left = RBnode(1, True, parent=tree2.root)
        tree2.root.right = RBnode(3, True, parent=tree2.root)
        tree2.remove(tree2.root, 1)

        self.assertEqual(tree2.root.value, 2)
        self.assertEqual(tree2.root.is_red, False)
        self.assertEqual(tree2.root.left, None)
        self.assertEqual(tree2.root.right.value, 3)
        self.assertEqual(tree2.root.right.is_red, True)

        # remove right child
        tree3 = RBtree()
        tree3.root = RBnode(5, False)
        tree3.root.left = RBnode(4, True, parent=tree3.root)
        tree3.root.right = RBnode(6, True, parent=tree3.root)
        tree3.remove(tree3.root, 6)

        self.assertEqual(tree3.root.value, 5)
        self.assertEqual(tree3.root.is_red, False)
        self.assertEqual(tree3.root.left.value, 4)
        self.assertEqual(tree3.root.left.is_red, True)
        self.assertEqual(tree3.root.right, None)

        # remove root only left child
        tree4 = RBtree()
        tree4.root = RBnode(8, False)
        tree4.root.left = RBnode(7, True, parent=tree4.root)
        tree4.remove(tree4.root, 8)

        self.assertEqual(tree4.root.value, 7)
        self.assertEqual(tree4.root.is_red, False)
        self.assertEqual(tree4.root.parent, None)
        self.assertEqual(tree4.root.left, None)
        self.assertEqual(tree4.root.right, None)

        # remove root only right child
        tree5 = RBtree()
        tree5.root = RBnode(11, False)
        tree5.root.right = RBnode(12, True, parent=tree5.root)
        tree5.remove(tree5.root, 11)

        self.assertEqual(tree5.root.value, 12)
        self.assertEqual(tree5.root.is_red, False)
        self.assertEqual(tree5.root.parent, None)
        self.assertEqual(tree5.root.left, None)
        self.assertEqual(tree5.root.right, None)

        # remove root two children
        tree6 = RBtree()
        tree6.root = RBnode(14, False)
        tree6.root.left = RBnode(13, True, parent=tree6.root)
        tree6.root.right = RBnode(15, True, parent=tree6.root)
        tree6.remove(tree6.root, 14)

        self.assertEqual(tree6.root.value, 13)
        self.assertEqual(tree6.root.is_red, False)
        self.assertEqual(tree6.root.parent, None)
        self.assertEqual(tree6.root.left, None)
        self.assertEqual(tree6.root.right.value, 15)

    def test_remove_medium(self):
        # internal node, two children
        tree = RBtree()
        tree.root = RBnode(15, False)
        tree.root.left = RBnode(8, False, parent=tree.root)
        tree.root.left.left = RBnode(2, True, parent=tree.root.left)
        tree.root.left.right = RBnode(11, True, parent=tree.root.left)
        tree.root.right = RBnode(21, False, parent=tree.root)
        tree.root.right.left = RBnode(18, True, parent=tree.root.right)
        tree.root.right.right = RBnode(25, True, parent=tree.root.right)

        tree.remove(tree.root, 8)
        
        assert tree.root.subtree_redblack_property()
        assert [node.value for node in tree.inorder(tree.root)] == [2, 11, 15, 18, 21, 25]

        # internal node, one child
        tree = RBtree()
        tree.root = RBnode(15, False)
        tree.root.left = RBnode(8, False, parent=tree.root)
        tree.root.left.right = RBnode(11, True, parent=tree.root.left)
        tree.root.right = RBnode(22, False, parent=tree.root)
        tree.root.right.left = RBnode(18, True, parent=tree.root.right)
        tree.root.right.right = RBnode(25, True, parent=tree.root.right)

        tree.remove(tree.root, 8)
        
        assert tree.root.subtree_redblack_property()
        assert [node.value for node in tree.inorder(tree.root)] == [11, 15, 18, 22, 25]

        # red external node
        tree = RBtree()
        tree.root = RBnode(15, False)
        tree.root.left = RBnode(8, False, parent=tree.root)
        tree.root.right = RBnode(23, False, parent=tree.root)
        tree.root.right.left = RBnode(17, True, parent=tree.root.right)
        tree.root.right.right = RBnode(25, True, parent=tree.root.right)

        tree.remove(tree.root, 8)

        assert tree.root.subtree_redblack_property()
        assert [node.value for node in tree.inorder(tree.root)] == [15, 17, 23, 25]

        # red internal with one grandchild
        tree = RBtree()
        tree.root = RBnode(14, False)
        tree.root.left = RBnode(8, False, parent=tree.root)
        tree.root.right = RBnode(22, True, parent=tree.root)
        tree.root.right.left = RBnode(17, False, parent=tree.root.right)
        tree.root.right.right = RBnode(25, False, parent=tree.root.right)
        tree.root.right.right.right = RBnode(28, True, parent=tree.root.right)

        tree.remove(tree.root, 22)

        assert tree.root.subtree_redblack_property()
        assert [node.value for node in tree.inorder(tree.root)] == [8, 14, 17, 25, 28]

        # black external with large sibling tree
        tree = RBtree()
        tree.root = RBnode(14, False)
        tree.root.left = RBnode(9, False, parent=tree.root)
        tree.root.right = RBnode(22, True, parent=tree.root)
        tree.root.right.left = RBnode(19, False, parent=tree.root.right)
        tree.root.right.right = RBnode(25, False, parent=tree.root.right)
        tree.root.right.right.right = RBnode(28, True, parent=tree.root.right)

        tree.remove(tree.root, 9)

        assert tree.root.subtree_redblack_property()
        assert [node.value for node in tree.inorder(tree.root)] == [14, 19, 22, 25, 28]

        # red internal with four grandchildren
        tree = RBtree()
        tree.root = RBnode(15, False)
        tree.root.left = RBnode(8, False, parent=tree.root)
        tree.root.right = RBnode(22, True, parent=tree.root)
        tree.root.right.left = RBnode(17, False, parent=tree.root.right)
        tree.root.right.right = RBnode(25, False, parent=tree.root.right)
        tree.root.right.left.left = RBnode(16, True, parent=tree.root.right.left)
        tree.root.right.left.right = RBnode(20, True, parent=tree.root.right.left)
        tree.root.right.right.left = RBnode(24, True, parent=tree.root.right.right)
        tree.root.right.right.right = RBnode(45, True, parent=tree.root.right.right)

        tree.remove(tree.root, 22)

        assert tree.root.subtree_redblack_property()
        assert [node.value for node in tree.inorder(tree.root)] == [8, 15, 16, 17, 20, 24, 25, 45]

        tree.remove(tree.root, 20)

        assert tree.root.subtree_redblack_property()
        assert [node.value for node in tree.inorder(tree.root)] == [8, 15, 16, 17, 24, 25, 45]

        tree.remove(tree.root, 25)

        assert tree.root.subtree_redblack_property()
        assert [node.value for node in tree.inorder(tree.root)] == [8, 15, 16, 17, 24, 45]

        tree.remove(tree.root, 15)

        assert tree.root.subtree_redblack_property()
        assert [node.value for node in tree.inorder(tree.root)] == [8, 16, 17, 24, 45]

        tree.remove(tree.root, 17)

        assert tree.root.subtree_redblack_property()
        assert [node.value for node in tree.inorder(tree.root)] == [8, 16, 24, 45]

if __name__ == '__main__':
    unittest.main()
