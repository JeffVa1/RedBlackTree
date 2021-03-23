# DO NOT MODIFY
from __future__ import annotations
from typing import TypeVar, Generic, Tuple

T = TypeVar('T')


class RBnode:
    """
    A red/black tree node class
    :value: stored value, also serves as key. Generic Type.
    :is_red: boolean identifier
    :parent: reference to parent node
    :left: reference to left node
    :right: reference to right node
    """

    __slots__ = ['value', 'is_red', 'parent', 'left', 'right']

    def __init__(self, value: Generic[T], is_red: bool = True, parent: RBnode = None,
                 left: RBnode = None, right: RBnode = None) -> None:
        """ Node Initializer"""
        self.value = value
        self.is_red = is_red
        self.parent = parent
        self.left, self.right = left, right

    def __eq__(self, other: RBnode) -> bool:
        """ Node Equality Comparator """
        return (type(None if not self else self.value) is type(None if not other else other.value) and
               self.value == other.value and
               self.is_red == other.is_red)

    def __str__(self) -> str:
        """ returns string representation of a node """
        r = '(R)' if self.is_red else '(B)'
        return str(self.value) + ' ' + r

    def __repr__(self) -> str:
        return self.__str__()

    def subtree_size(self) -> int:
        """ returns size of tree rooted at given node """
        return 1 + (self.left.subtree_size() if self.left else 0) + (self.right.subtree_size() if self.right else 0)

    def subtree_height(self) -> int:
        """ returns the height of a subtree rooted at a given node """
        return 1 + max(self.left.subtree_height() if self.left else 0, self.right.subtree_height() if self.right else 0)

    def subtree_redblack_property(self) -> bool:
        """ returns whether a tree adheres to red black properties """

        def rb_check_helper(node) -> Tuple[bool, int]:
            """ recursive helper """
            if not node:
                return True, 1
            if not node.parent and node.is_red:
                return False, 0
            if node.is_red and ((node.left and node.left.is_red) or (node.right and node.right.is_red)):
                return False, -1

            left_check, num_black_left = rb_check_helper(node.left)
            right_check, num_black_right = rb_check_helper(node.right)
            return all([left_check, right_check, num_black_left == num_black_right]), num_black_left + (0 if node.is_red else 1)

        return rb_check_helper(self)[0]
