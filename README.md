# Blind75_NC_TUF

TEMPLATES FOR COMMON PATTERNS :

Here are some common LeetCode patterns along with their general templates in Java:
1. Two-pointer Pattern (Sorted Array / String)
    Ways to identify when to use the Two Pointer method:
    •	It will feature problems where you deal with sorted arrays (or Linked Lists) and need to find a set of elements that fulfill 
        certain constraints
    •	The set of elements in the array is a pair, a triplet, or even a subarray

    
    Problem types:
    •	Finding pairs that sum to a target.
    •	Checking if a string is a palindrome.
    •	Merging sorted arrays.

    public int[] twoPointerTemplate(int[] arr, int target) {
        Arrays.sort(arr); // Sort if necessary
    
        int left = 0;
        int right = arr.length - 1;
    
        while (left < right) {
            int sum = arr[left] + arr[right];
    
            if (sum == target) {
                return new int[] { arr[left], arr[right] };
            } else if (sum < target) {
                left++; // Move the left pointer
            } else {
                right--; // Move the right pointer
            }
        }
    
        return null; // Return null if no pair is found
    }
 
2. Sliding Window Pattern
   Following are some ways you can identify that the given problem might require a sliding window:
    •	The problem input is a linear data structure such as a linked list, array, or string
    •	You’re asked to find the longest/shortest substring, subarray, or a desired value
   
   Problem types:
    •	Finding the maximum sum of subarrays of size k.
    •	Longest substring without repeating characters.
    •	Smallest subarray with a given sum.
    
    public int slidingWindowTemplate(int[] arr, int k) {
        int maxSum = 0, currentSum = 0;
    
        for (int i = 0; i < k; i++) {
            currentSum += arr[i]; // Sum first 'k' elements
        }
    
        maxSum = currentSum;
    
        for (int i = k; i < arr.length; i++) {
            currentSum += arr[i] - arr[i - k]; // Slide the window
            maxSum = Math.max(maxSum, currentSum); // Update max sum
        }
    
        return maxSum;
    }

4. Binary Search Pattern
Problem types:
•	Searching for an element in a sorted array.
•	Finding the first/last occurrence of a target.
•	Searching in a rotated sorted array.

public int binarySearchTemplate(int[] arr, int target) {
    int left = 0, right = arr.length - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2; // Prevents overflow

        if (arr[mid] == target) {
            return mid; // Target found
        } else if (arr[mid] < target) {
            left = mid + 1; // Search in the right half
        } else {
            right = mid - 1; // Search in the left half
        }
    }

    return -1; // Target not found
}

4. Fast & Slow Pointer Pattern (Floyd’s Cycle Detection)
    How do you identify when to use the Fast and Slow pattern?
    •	The problem will deal with a loop in a linked list or array
    •	When you need to know the position of a certain element or the overall length of the linked list.
    When should I use it over the Two Pointer method mentioned above?
    •	There are some cases where you shouldn’t use the Two Pointer approach such as in a singly linked list where you can’t move in a backwards direction. An example of when to use the Fast and Slow pattern is when you’re trying to determine if a linked list is a palindrome.

    Problem types:
    •	Detecting a cycle in a linked list.
    •	Finding the middle of a linked list.
    •	Finding the starting point of the cycle.
    
    public boolean hasCycle(ListNode head) {
        if (head == null || head.next == null) return false;
    
        ListNode slow = head;
        ListNode fast = head;
    
        while (fast != null && fast.next != null) {
            slow = slow.next; // Move slow pointer by 1 step
            fast = fast.next.next; // Move fast pointer by 2 steps
    
            if (slow == fast) {
                return true; // Cycle detected
            }
        }
    
        return false; // No cycle
    }

5. Merge Intervals Pattern
How do you identify when to use the Merge Intervals pattern?
•	If you’re asked to produce a list with only mutually exclusive intervals
•	If you hear the term “overlapping intervals”.
Merge interval problem patterns:
•	Intervals Intersection (medium)
•	Maximum CPU Load (hard)

Problem types:
•	Merging overlapping intervals.
•	Inserting an interval into a list of intervals.
•	Checking if two intervals overlap.

public int[][] mergeIntervals(int[][] intervals) {
    if (intervals.length <= 1) return intervals;

    // Sort intervals by the start time
    Arrays.sort(intervals, (a, b) -> Integer.compare(a[0], b[0]));

    List<int[]> result = new ArrayList<>();
    int[] currentInterval = intervals[0];

    for (int i = 1; i < intervals.length; i++) {
        if (intervals[i][0] <= currentInterval[1]) {
            // Overlapping intervals, merge them
            currentInterval[1] = Math.max(currentInterval[1], intervals[i][1]);
        } else {
            // Non-overlapping interval, add to result and move to the next
            result.add(currentInterval);
            currentInterval = intervals[i];
        }
    }

    result.add(currentInterval); // Add the last interval
    return result.toArray(new int[result.size()][]);
}

6. Backtracking Pattern
Problem types:
•	Generating all subsets of a set.
•	Solving the N-Queens problem.
•	Permutations and combinations.

public List<List<Integer>> backtrackingTemplate(int[] nums) {
    List<List<Integer>> result = new ArrayList<>();
    backtrack(result, new ArrayList<>(), nums, 0);
    return result;
}

private void backtrack(List<List<Integer>> result, List<Integer> tempList, int[] nums, int start) {
    result.add(new ArrayList<>(tempList)); // Add current subset to result

    for (int i = start; i < nums.length; i++) {
        tempList.add(nums[i]); // Include the element
        backtrack(result, tempList, nums, i + 1); // Recurse with next index
        tempList.remove(tempList.size() - 1); // Remove the last element (backtrack)
    }
}
 
7. Dynamic Programming (DP) Pattern
Problem types:
•	Finding the nth Fibonacci number.
•	Knapsack problems.
•	Longest common subsequence.

public int dynamicProgrammingTemplate(int n) {
    if (n <= 1) return n;

    int[] dp = new int[n + 1];
    dp[0] = 0;
    dp[1] = 1;

    for (int i = 2; i <= n; i++) {
        dp[i] = dp[i - 1] + dp[i - 2]; // Example for Fibonacci sequence
    }

    return dp[n];
}
 
8. Union-Find Pattern (Disjoint Set Union - DSU)
Problem types:
•	Finding connected components in a graph.
•	Detecting cycles in an undirected graph.
•	Kruskal’s algorithm for minimum spanning tree.

class UnionFind {
    private int[] parent;
    private int[] rank;

    public UnionFind(int n) {
        parent = new int[n];
        rank = new int[n];
        for (int i = 0; i < n; i++) {
            parent[i] = i; // Initially, each element is its own parent
        }
    }

    public int find(int x) {
        if (x != parent[x]) {
            parent[x] = find(parent[x]); // Path compression
        }
        return parent[x];
    }

    public boolean union(int x, int y) {
        int rootX = find(x);
        int rootY = find(y);

        if (rootX != rootY) {
            if (rank[rootX] > rank[rootY]) {
                parent[rootY] = rootX;
            } else if (rank[rootX] < rank[rootY]) {
                parent[rootX] = rootY;
            } else {
                parent[rootY] = rootX;
                rank[rootX]++;
            }
            return true;
        }
        return false;
    }
}
 
These templates cover common problem-solving patterns found in many LeetCode problems and can be adapted to a wide range of challenges.

![image](https://github.com/user-attachments/assets/619e0f8d-16b7-4995-a126-2ad6b454a742)



TRICKS -

Here are some common DSA patterns and tricks implemented in Java:
1. Sliding Window Pattern
•	Use Case: Problems that involve finding a subarray or substring that meets specific conditions, such as finding the maximum sum of a subarray of size k, or longest substring with unique characters.
•	Trick: Maintain a window with two pointers or indices, and adjust the window size based on problem constraints.
Example: Maximum sum of a subarray of size k:

public class SlidingWindow {
    public static int maxSumSubarray(int[] arr, int k) {
        int maxSum = 0, windowSum = 0;

        // Calculate the sum of the first window of size k
        for (int i = 0; i < k; i++) {
            windowSum += arr[i];
        }

        // Slide the window over the array
        for (int i = k; i < arr.length; i++) {
            windowSum += arr[i] - arr[i - k];  // Slide the window right
            maxSum = Math.max(maxSum, windowSum);  // Update the maximum sum
        }

        return maxSum;
    }

    public static void main(String[] args) {
        int[] arr = {2, 1, 5, 1, 3, 2};
        int k = 3;
        System.out.println("Maximum sum of a subarray of size " + k + ": " + maxSumSubarray(arr, k));
    }
}
2. Two Pointers Pattern
•	Use Case: Typically used for problems like finding pairs that meet certain conditions, or when the array is sorted, such as the "Two Sum" problem or the "Container with Most Water" problem.
•	Trick: Use two pointers, one starting from the beginning and the other from the end, and move them towards each other based on conditions.
Example: Two Sum problem with sorted array:

public class TwoPointers {
    public static int[] twoSumSorted(int[] nums, int target) {
        int left = 0, right = nums.length - 1;

        while (left < right) {
            int sum = nums[left] + nums[right];
            if (sum == target) {
                return new int[] {left, right};  // Return the indices of the two numbers
            } else if (sum < target) {
                left++;  // Move the left pointer to the right
            } else {
                right--;  // Move the right pointer to the left
            }
        }

        return new int[] {-1, -1};  // Return -1 if no valid pair is found
    }

    public static void main(String[] args) {
        int[] nums = {1, 2, 3, 4, 6};
        int target = 6;
        int[] result = twoSumSorted(nums, target);
        System.out.println("Indices of the two numbers: " + result[0] + ", " + result[1]);
    }
}
3. Fast and Slow Pointers (Tortoise and Hare)
•	Use Case: Used for cyclic linked list problems, finding cycles, or determining if a number sequence has a cycle (e.g., detecting loops in a linked list).
•	Trick: Use two pointers, one moving twice as fast as the other. If they meet, a cycle exists.
Example: Detecting a cycle in a linked list:

class ListNode {
    int val;
    ListNode next;

    ListNode(int val) {
        this.val = val;
        this.next = null;
    }
}

public class CycleDetection {
    public static boolean hasCycle(ListNode head) {
        if (head == null || head.next == null) {
            return false;
        }

        ListNode slow = head;
        ListNode fast = head;

        while (fast != null && fast.next != null) {
            slow = slow.next;  // Move slow by 1 step
            fast = fast.next.next;  // Move fast by 2 steps

            if (slow == fast) {  // If both meet, there's a cycle
                return true;
            }
        }

        return false;
    }

    public static void main(String[] args) {
        ListNode head = new ListNode(3);
        head.next = new ListNode(2);
        head.next.next = new ListNode(0);
        head.next.next.next = new ListNode(-4);
        head.next.next.next.next = head.next;  // Creating a cycle

        System.out.println("Does the linked list have a cycle? " + hasCycle(head));
    }
}
4. Merge Intervals Pattern
•	Use Case: Problems that involve intervals, such as merging overlapping intervals, finding gaps between intervals, or finding if a new interval conflicts with existing ones.
•	Trick: Sort the intervals by the start time and merge overlapping intervals as you iterate.
Example: Merging overlapping intervals:

import java.util.*;

public class MergeIntervals {
    public static int[][] merge(int[][] intervals) {
        if (intervals.length <= 1) {
            return intervals;
        }

        // Sort intervals based on the starting time
        Arrays.sort(intervals, (a, b) -> Integer.compare(a[0], b[0]));

        List<int[]> merged = new ArrayList<>();
        int[] currentInterval = intervals[0];
        merged.add(currentInterval);

        for (int[] interval : intervals) {
            int currentEnd = currentInterval[1];
            int nextStart = interval[0];
            int nextEnd = interval[1];

            if (currentEnd >= nextStart) {  // Overlapping intervals, merge them
                currentInterval[1] = Math.max(currentEnd, nextEnd);
            } else {  // No overlap, move to the next interval
                currentInterval = interval;
                merged.add(currentInterval);
            }
        }

        return merged.toArray(new int[merged.size()][]);
    }

    public static void main(String[] args) {
        int[][] intervals = {{1, 3}, {2, 6}, {8, 10}, {15, 18}};
        int[][] result = merge(intervals);
        System.out.println("Merged intervals: ");
        for (int[] interval : result) {
            System.out.println(Arrays.toString(interval));
        }
    }
}
5. Backtracking Pattern
•	Use Case: Solving problems like generating permutations, combinations, and subsets, or solving puzzles like N-Queens and Sudoku.
•	Trick: Try all possibilities recursively, backtrack when a condition is not met, and undo changes before moving to the next possibility.
Example: Generating all subsets of a set:

import java.util.*;

public class Subsets {
    public static List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        generateSubsets(0, nums, new ArrayList<>(), result);
        return result;
    }

    private static void generateSubsets(int index, int[] nums, List<Integer> current, List<List<Integer>> result) {
        result.add(new ArrayList<>(current));  // Add the current subset to the result

        for (int i = index; i < nums.length; i++) {
            current.add(nums[i]);  // Choose an element
            generateSubsets(i + 1, nums, current, result);  // Explore further
            current.remove(current.size() - 1);  // Backtrack and remove the element
        }
    }

    public static void main(String[] args) {
        int[] nums = {1, 2, 3};
        List<List<Integer>> subsets = subsets(nums);
        System.out.println("All subsets: " + subsets);
    }
}
Conclusion
These are some of the fundamental patterns used in solving DSA problems. Recognizing these patterns can help streamline problem-solving in coding interviews and competitions. Each pattern has its own tricks and can often be applied across multiple problems.

![image](https://github.com/user-attachments/assets/2ec3f1cd-b8d4-42e2-b0e6-6bdba8afd950)
