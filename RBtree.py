"""
Project 3 (Fall 2020) - Red/Black Trees
Name: Jeffrey Valentic
Professor Onsay
10/27/2020
"""

from __future__ import annotations
from typing import TypeVar, Generic, Callable, Generator
from Project3.RBnode import RBnode as Node
from copy import deepcopy
import queue

T = TypeVar('T')

class RBtree:
    """
    A Red/Black Tree class
    :root: Root Node of the tree
    :size: Number of Nodes
    """

    __slots__ = ['root', 'size']

    def __init__(self, root: Node = None):
        """ Initializer for an RBtree """
        # this alllows us to initialize by copying an existing tree
        self.root = deepcopy(root)
        if self.root:
            self.root.parent = None
        self.size = 0 if not self.root else self.root.subtree_size()

    def __eq__(self, other: RBtree) -> bool:
        """ Equality Comparator for RBtrees """
        comp = lambda n1, n2: n1 == n2 and (
            (comp(n1.left, n2.left) and comp(n1.right, n2.right)) if (n1 and n2) else True)
        return comp(self.root, other.root) and self.size == other.size

    def __str__(self) -> str:
        """ represents Red/Black tree as string """

        if not self.root:
            return 'Empty RB Tree'

        root, bfs_queue, height = self.root, queue.SimpleQueue(), self.root.subtree_height()
        track = {i: [] for i in range(height + 1)}
        bfs_queue.put((root, 0, root.parent))

        while bfs_queue:
            n = bfs_queue.get()
            if n[1] > height:
                break
            track[n[1]].append(n)
            if n[0] is None:
                bfs_queue.put((None, n[1] + 1, None))
                bfs_queue.put((None, n[1] + 1, None))
                continue
            bfs_queue.put((None, n[1] + 1, None) if not n[0].left else (n[0].left, n[1] + 1, n[0]))
            bfs_queue.put((None, n[1] + 1, None) if not n[0].right else (n[0].right, n[1] + 1, n[0]))

        spaces = 12 * (2 ** (height))
        ans = '\n' + '\t\tVisual Level Order Traversal of RBtree'.center(spaces) + '\n\n'
        for i in range(height):
            ans += f"Level {i + 1}: "
            for n in track[i]:
                space = int(round(spaces / (2 ** i)))
                if not n[0]:
                    ans += ' ' * space
                    continue
                ans += "{} ({})".format(n[0], n[2].value if n[2] else None).center(space, " ")
            ans += '\n'
        return ans

    def __repr__(self) -> str:
        return self.__str__()

    ################################################################
    ################### Complete Functions Below ###################
    ################################################################

    ######################## Static Methods ########################
    # These methods are static as they operate only on nodes, without explicitly referencing an RBtree instance

    @staticmethod
    def set_child(parent: Node, child: Node, is_left: bool) -> None:
        """
        ARGS:    parent - parent who will recieve new child
                  child - node to be set as child
                is_left - if true, set child as parent.left child
        Return:    None
        This function sets the parents child to the child node, using is_left to
        determine which side to set as the child.
        """
        if parent.is_red:
            child.is_red = False
        else:
            child.is_red = True

        if is_left:
            parent.left = child
        else:
            parent.right = child
        child.parent = parent

    @staticmethod
    def replace_child(parent: Node, current_child: Node, new_child: Node) -> None:
        """
        ARGS:      parent - parent of node to replace
            current_child - node to be replaced
                new_child - node replacing current_child
        Return:      None
        This function replaces the current_child with new_child and reassigns pointers in RBTree.
        """
        new_child.is_red = current_child.is_red
        if parent.left == current_child:
            parent.left = new_child
            new_child.parent = parent
        else:
            parent.right = new_child
            new_child.parent = parent

    @staticmethod
    def get_sibling(node: Node) -> Node:
        """
        ARGS:   Node - node whose sibiling is to be found
        Return: Node - sibiling node of Node input
        This function locates and returns the sibiling of the node
        passed as an argument to this function.
        """
        if node.parent is None:
            return None
        else:
            if node.parent.left == node:
                if node.parent.right is not None:
                    return node.parent.right
                else:
                    return None
            else:
                if node.parent.left is not None:
                    return node.parent.left
                else:
                    return None

    @staticmethod
    def get_grandparent(node: Node) -> Node:
        """
        ARGS:   Node - node whose grandparent is to be found
        Return: Node - grandparent node of Node input - parents parent
        This function locates and returns the parent of the parent
        of the node passed as an argument to this function (ie the grandparent).
        """
        if node.parent is not None and node.parent.parent is not None:
            return node.parent.parent
        else:
            return None

    @staticmethod
    def get_uncle(node: Node) -> Node:
        """
        ARGS:   Node - node whose uncle is to be found
        Return: Node - uncle node of Node input - parents sibiling
        This function locates and returns the sibiling of the parent
        of the node passed as an argument to this function.
        """
        grandparent = None
        if node.parent is not None:
            grandparent = node.parent.parent
        if grandparent is None:
            return None
        if grandparent.left is node.parent:
            return grandparent.right
        else:
            return grandparent.left


    ######################## Misc Utilities ##########################

    def min(self, node: Node) -> Node:
        """
        ARGS:   self
                Node - Root of subtree
        Return: Node - node with min value in subtree
        This function recursively searches through a RBTree rooted at Node to locate and return a
        node with the minimum value in that subtree.
        """

        def min_helper(cur_node: Node) -> Node:
            if self.root is None:
                return None
            if cur_node.left is not None:
                return min_helper(cur_node.left)
            return cur_node

        return min_helper(node)

    def max(self, node: Node) -> Node:
        """
        ARGS:   self
                Node - Root of subtree
        Return: Node - node with max value in subtree
        This function recursively searches through a RBTree rooted at Node to locate and return a
        node with the maximum value in that subtree.
        """

        def max_helper(cur_node: Node) -> Node:
            if self.root is None:
                return None
            if cur_node.right is not None:
                return max_helper(cur_node.right)
            return cur_node

        return max_helper(node)

    def search(self, node: Node, val: Generic[T]) -> Node:
        """
        ARGS:   self
                Node - Root of subtree
                 val - value to look for in subtree
        Return: Node - node with value val or None if node not found)
        This function recursively searches through a RBTree rooted at Node to locate and return a
        node with a value that matches val.
        """

        def search_helper(cur_node: Node) -> Node:
            if val < cur_node.value:
                if cur_node.left is not None:
                    return search_helper(cur_node.left)
                else:
                    return cur_node
            if val > cur_node.value:
                if cur_node.right is not None:
                    return search_helper(cur_node.right)
                else:
                    return cur_node
            return cur_node

        if node is None:
            return node
        if node.right is None and node.left is None:
            return node
        return search_helper(node)

    ######################## Tree Traversals #########################

    def inorder(self, node: Node) -> Generator[Node, None, None]:
        """
        ARGS:        self
                     Node - root of subtree to evaluate
        Return: Generator - A inorder representation of the RBTree rooted at Node.
        This function evaluates the subtree with root Node to create a Generator that contains the
        inorder sequence of the subtree.
        """

        def inorder_helper(cur_node: Node):
            if cur_node:
                yield from inorder_helper(cur_node.left)
                yield cur_node
                yield from inorder_helper(cur_node.right)

        return inorder_helper(node)

    def preorder(self, node: Node) -> Generator[Node, None, None]:
        """
        ARGS:        self
                     Node - root of subtree to evaluate
        Return: Generator - A preorder representation of the RBTree rooted at Node.
        This function evaluates the subtree with root Node to create a Generator that contains the
        preorder sequence of the subtree.
        """

        def preorder_helper(cur_node: Node):
            if cur_node:
                yield cur_node
                yield from preorder_helper(cur_node.left)
                yield from preorder_helper(cur_node.right)

        return preorder_helper(node)

    def postorder(self, node: Node) -> Generator[Node, None, None]:
        """
        ARGS:        self
                     Node - root of subtree to evaluate
        Return: Generator - A postorder representation of the RBTree rooted at Node.
        This function evaluates the subtree with root Node to create a Generator that contains the
        postorder sequence of the subtree.
        """

        def postorder_helper(cur_node: Node):
            if cur_node:
                if cur_node.left is not None:
                    yield from postorder_helper(cur_node.left)

                if cur_node.right is not None:
                    yield from postorder_helper(cur_node.right)
                yield cur_node

        return postorder_helper(node)

    def bfs(self, node: Node) -> Generator[Node, None, None]:
        """
        ARGS:        self
                     Node - root of subtree to evaluate
        Return: Generator - A Breadth First representation of the RBTree rooted at Node.
        This function evaluates the subtree with root Node to create a Generator that contains the breadth first
        order of the subtree.
        """
        q = queue.SimpleQueue()
        q.put(node)

        while not q.empty():
            cur_node = q.get()
            if cur_node:
                if cur_node.left:
                    q.put(cur_node.left)
                if cur_node.right:
                    q.put(cur_node.right)
                yield cur_node

    ################### Rebalancing Utilities ######################

    def left_rotate(self, node: Node) -> None:
        """
        ARGS:   self
                Node - node to rotate around)
        Return: None
        This function rotates a node and its children in the counter-clockwise direction.
        It adjusts red-black properties and balancing during rotation.
        """
        temp_node = node.right
        node.right = temp_node.left
        if temp_node.left is not None:
            temp_node.left.parent = node
        temp_node.parent = node.parent

        if node.parent is None:
            self.root = temp_node
        elif node is node.parent.right:
            node.parent.right = temp_node
        elif node is node.parent.left:
            node.parent.left = temp_node

        temp_node.left = node
        node.parent = temp_node

    def right_rotate(self, node: Node) -> None:
        """
        ARGS:   self
                Node - node to rotate around
        Return: None
        This function rotates a node and its children in the clockwise direction.
        It adjusts red-black properties and balancing during rotation.
        """
        temp_node = node.left
        node.left = temp_node.right
        if temp_node.right is not None:
            temp_node.right.parent = node
        temp_node.parent = node.parent

        if node.parent is None:
            self.root = temp_node
        elif node is node.parent.left:
            node.parent.left = temp_node
        elif node is node.parent.right:
            node.parent.right = temp_node

        temp_node.right = node
        node.parent = temp_node

    def insertion_repair(self, node: Node) -> None:
        """
        ARGS:   self
                Node - node of subtree to repair
        Return: None
        This function recursively repairs damage to red-black property and balance of the RBTree after
        a new node has been inserted.
        """
        def first_checks(cur_node: Node) -> bool:
            if cur_node.parent is None:
                cur_node.is_red = False
                return False
            if not cur_node.parent.is_red:
                return False
            else:
                return True

        def ir_helper(cur_node: Node):
            if first_checks(cur_node):

                if cur_node.parent.is_red:
                    parent = cur_node.parent
                    grandparent = self.get_grandparent(cur_node)
                    uncle = self.get_uncle(cur_node)

                    if uncle is not None and uncle.is_red:
                        parent.is_red = uncle.is_red = False
                        grandparent.is_red = True
                        self.insertion_repair(grandparent)
                        return

                    if cur_node is parent.right and parent is grandparent.left:
                        self.left_rotate(parent)
                        cur_node = parent
                        parent = cur_node.parent

                    elif cur_node is parent.left and parent is grandparent.right:
                        self.right_rotate(parent)
                        cur_node = parent
                        parent = cur_node.parent

                    parent.is_red = False
                    grandparent.is_red = True
                    if cur_node is parent.left:
                        self.right_rotate(grandparent)
                    elif cur_node is parent.right:
                        self.left_rotate(grandparent)

                self.root.is_red = False

        ir_helper(node)

    def prepare_removal(self, node: Node) -> None:
        """
        ARGS:   self
                Node - location to repair
        Return: None
        This function prepares a RBTree to have a node removed.
        It does this by evaluating the relationships present in the tree and making adjustments.
        The adjustments are made to ensure that the RBTree follows red-black properties and is balanced.
        """

        #CHECKS
        def both_children_black(cur_node: Node):
            if cur_node.left is not None and cur_node.left.is_red:
                return False
            if cur_node.right is not None and cur_node.right.is_red:
                return False
            return True

        def tree_null_or_black(cur_node: Node):
            if cur_node is None:
                return True
            return not cur_node.is_red

        def tree_non_nul_and_red(cur_node: Node):
            if cur_node is None:
                return False
            return cur_node.is_red

        # REMOVE CASES
        def remove_case1(cur_node: Node):
            if cur_node.is_red or cur_node.parent is None:
                return True
            else:
                return False

        def remove_case2(cur_node: Node, sibling: Node):
            if sibling.is_red:
                cur_node.parent.is_red = True
                sibling.is_red = False
                if cur_node is cur_node.parent.left:
                    self.left_rotate(cur_node.parent)
                else:
                    self.right_rotate(cur_node.parent)

                return True
            return False

        def remove_case3(cur_node: Node, sibling: Node):
            if not cur_node.parent.is_red and both_children_black(sibling):
                sibling.is_red = True
                self.prepare_removal(cur_node.parent)
                return True
            return False

        def remove_case4(cur_node: Node, sibling: Node):
            if cur_node.parent.is_red and both_children_black(sibling):
                cur_node.parent.is_red = False
                sibling.is_red = True
                return True
            return False

        def remove_case5(cur_node: Node, sibling: Node):
            if tree_non_nul_and_red(sibling.left) and tree_null_or_black(sibling.right):
                if cur_node is cur_node.parent.left:
                    sibling.is_red = True
                    sibling.left.is_red = False
                    self.right_rotate(sibling)
                    return True
            return False

        def remove_case6(cur_node: Node, sibling: Node):
            if tree_null_or_black(sibling.left) and tree_non_nul_and_red(sibling.right):
                if cur_node is cur_node.parent.right:
                    sibling.is_red = True
                    sibling.right.is_red = False
                    self.left_rotate(sibling)
                    return True
            return False

        if remove_case1(node):
            return

        sibling = self.get_sibling(node)
        if remove_case2(node, sibling):
            sibling = self.get_sibling(node)

        if remove_case3(node, sibling):
            return

        if remove_case4(node, sibling):
            return

        if remove_case5(node, sibling):
            sibling = self.get_sibling(node)

        if remove_case6(node, sibling):
            sibling = self.get_sibling(node)

        sibling.is_red = node.parent.is_red
        node.parent.is_red = False

        if node is node.parent.left:
            sibling.right.is_red = False
            self.left_rotate(node.parent)
        else:
            sibling.left.is_red = False
            self.right_rotate(node.parent)

    ##################### Insertion and Removal #########################

    def insert(self, node: Node, val: Generic[T]) -> None:
        """
        ARGS:   self
                Node - root of subtree
                val - value of new node to insert
        Return: None
        This function uses recursion to insert a new node in the proper position in a subtree.
        The insertion_repair function is called in order to maintain balance and red-black properties of the tree.
        """
        new_node = Node(val)
        if self.root is None:
            self.root, self.root.value, self.root.is_red = new_node, val, False
            new_node.parent = None
            new_node.left = None
            new_node.right = None
            return

        def insert_helper(cur_node: Node, new_node: Node):
            if cur_node is not None:

                # adding the new node to the left side of the tree
                if new_node.value < cur_node.value:
                    if cur_node.left is None:
                        cur_node.left = new_node
                        new_node.parent = cur_node
                        cur_node = None
                    else:
                        insert_helper(cur_node.left, new_node)
                # adding the new node to the right side of the tree
                else:
                    if cur_node.right is None:
                        cur_node.right = new_node
                        new_node.parent = cur_node
                        cur_node = None
                    else:
                        insert_helper(cur_node.right, new_node)
            new_node.left = None
            new_node.right = None

        existing = self.search(self.root, new_node.value)
        if existing.value != new_node.value:
            insert_helper(node, new_node)
            new_node.is_red = True
            self.insertion_repair(new_node)
            self.size = self.root.subtree_size()

    def remove(self, node: Node, val: Generic[T]) -> None:
        """
        ARGS:   self
                Node - root of subtree
                val - node value to remove from subtree
        Return: None
        This function uses recursion and bst_remove to remove a node with a specified value.
        Before node is removed, prepare_removal is called to ensure the tree maintains balance.
        """
        def bst_remove(val):
            node_found = self.search(self.root, val)
            #CASE if search is unsuccessful
            if node_found is None or node_found.value != val:
                return
            #CASE search is successful, delete node
            #Leaf Node
            if node_found.right is None and node_found.left is None:
                par = node_found.parent
                #update parent
                if par is None:
                    self.root = None
                #if node was right or left child
                elif par.right is not None and par.right.value == val:
                    par.right = None
                else:
                    par.left = None
                self.size -= 1
                return

            #one child
            if node_found.right is None or node_found.left is None:
                par = node_found.parent
                if node_found.right is not None:
                    child = node_found.right
                else:
                    child = node_found.left

                #update parent (move single child to pos of current node)
                if par is None:
                    self.root = child
                    self.root.is_red = False
                elif par.right is not None and par.right.value == val:
                    par.right = child
                else:
                    par.left = child
                #update child
                child.parent = par
                self.size -= 1
                return

            #two children
            if node_found.left.is_red:
                min_node = self.min(node_found)
                self.prepare_removal(self.root)
                bst_remove(min_node.value)
                node_found.value = min_node.value
            else:
                min_node = self.min(node_found.right)
                self.prepare_removal(min_node)
                bst_remove(min_node.value)
                node_found.value = min_node.value

        def get_predecessor(cur_node: Node):
            cur_node = cur_node.left
            while cur_node.right is not None:
                cur_node = cur_node.right
            return cur_node

        cur_node = self.search(node, val)
        if cur_node is None:
            return

        if node.right is None and node.left is not None:
            pred = node.left
            pred_val = pred.value
            self.remove(pred, pred_val)
            node.value = pred_val
            return

        if cur_node.right is not None and cur_node.left is not None:
            pred = get_predecessor(cur_node)
            pred_val = pred.value
            self.remove(node, pred_val)
            cur_node.value = pred_val
            return

        if not cur_node.is_red:
            self.prepare_removal(cur_node)
        bst_remove(cur_node.value)



