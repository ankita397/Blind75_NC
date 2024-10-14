# Blind75_NC_TUF

TEMPLATES FOR COMMON PATTERNS :

1. Two-pointer Pattern (Sorted Array / String)
Problem types:

•	Finding pairs that sum to a target.
•	Checking if a string is a palindrome.
•	Merging sorted arrays.
•	Template:

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
Problem types:

•	Finding the maximum sum of subarrays of size k.
•	Longest substring without repeating characters.
•	Smallest subarray with a given sum.
•	Template:

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

3. Binary Search Pattern
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
