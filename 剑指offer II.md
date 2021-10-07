# 剑指offer II

## 整数

### [1. 整数除法](https://leetcode-cn.com/problems/xoh6Oh/)

感谢**老汤**的视频：https://leetcode-cn.com/problems/xoh6Oh/solution/jian-dan-yi-dong-javac-pythonjs-zheng-sh-e8r6/

```java
class Solution {
    public int divide(int a, int b) {
        //-2^31除以-1的结果会越界，所以在这种情况下需要返回2^31 - 1
        if (a == Integer.MIN_VALUE && b == -1) {
            return Integer.MAX_VALUE;
        }
        //这里对a和b的符号做了是否同号的判断：
        //若同号则结果为正数；不同号则为负数
        int sign = (a > 0) ^ (b > 0) ? -1 : 1;
        a = Math.abs(a);
        b = Math.abs(b);
        int res = 0;

        //要理解这里的实质：统计a可以被多少个b相加得到，b的个数 * sign就是要返回的结果
        //为了优化多个b相加的过程，可以用a - (b的倍数)来做优化
        //b的倍数可以用为运算的左移操作来优化
        //需要注意的是b的倍数可能会越界，那么我们换种思路让a右移，这就不会越界了，且道理是和b左移一样的

        // 无符号右移的目的是：将 -2147483648 看成 2147483648
        // 注意，这里不能是 (a >>> i) >= b 而应该是 (a >>> i) - b >= 0
        // 这个也是为了避免 b = -2147483648，如果 b = -2147483648
        // 那么 (a >>> i) >= b 永远为 true，但是 (a >>> i) - b >= 0 为 false
        for (int i = 31; i >= 0; i--) {
            if ((a >>> i) - b >= 0) {
                a -= (b << i);
                res += (1 << i);
            }
        }

        return sign == 1 ? res : -res;
    }
}
```



### [2. 二进制加法](https://leetcode-cn.com/problems/JFETK5/)

```java
class Solution {
    public String addBinary(String a, String b) {
        StringBuilder res = new StringBuilder();
        //carry用于存储当前的进位
        int carry = 0;
        int l1 = a.length() - 1;
        int l2 = b.length() - 1;
        while (l1 >= 0 || l2 >= 0) {
            //取出l1和l2指针指向的数位
            int x = l1 < 0 ? 0 : a.charAt(l1) - '0';
            int y = l2 < 0 ? 0 : b.charAt(l2) - '0';

            //计算两数位和之前计算的进位的和
            int sum = x + y + carry;
            //把结果拼接在res后，计算新的进位
            res.append(sum % 2);
            carry = sum / 2;

            //让两个指针往前移动
            l1--;
            l2--;
        }

        //循环结束后，若进位不为0，需要把剩余的进位拼接在res后
        if (carry != 0) {
            res.append(carry);
        }

        //循环是从尾到头扫描a、b的，所以res是反的，需要对res进行反转后再返回
        return res.reverse().toString();
    }
}
```

这题和[43.字符串相乘](https://leetcode-cn.com/problems/multiply-strings/)异曲同工，两题应该结合起来看



### [3. 前n个数字二进制中1的个数](https://leetcode-cn.com/problems/w3tCBm/)

```java
class Solution {
    public int[] countBits(int n) {
        int[] ans = new int[n + 1];

        for (int i = 1; i <= n; i++) {
            //若当前数i是偶数，可以发现当前数的二进制1的位数和i/2的二进制数是一样的
            //比如：2(10),4(100),8(1000)...每个偶数都是它的一半往左移一位
            if (i % 2 == 0) {
                ans[i] = ans[i / 2];
            } else {
                //若当前数i是奇数，那它的二进制1的个数是i-1的二进制1的个数+1
                ans[i] = ans[i - 1] + 1;
            }
        }

        return ans;
    }
}
```



### [4. 只出现一次的数字](https://leetcode-cn.com/problems/WGki4K/)

解法一：全用位运算，很难理解。抄了K神的答案，但我还是没搞太懂；或者说看懂了，还是没能学会其精髓

[k神的题解链接](https://leetcode-cn.com/problems/WGki4K/solution/jian-zhi-offer-ii-004-zhi-chu-xian-yi-ci-l3ud/)

```java
//这个解法的大体思路就是把每一位的二进制位的状态变化情况给列举出来（有限状态自动机）
//用位运算做了优化，特别是用与和非运算把if else都省了，这操作太强了吧
//需要ones和twos两个位的原因：对3取余有0、1、2三种状态，用一位是无法表示三种状态的
//ones的状态变化还能看懂，twos的状态变化看傻了
class Solution {
    public int singleNumber(int[] nums) {
        int ones = 0, twos = 0;
        for (int num : nums) {
            ones = ones ^ num & ~twos;
            twos = twos ^ num & ~ones;
        }
        return ones;
    }
}
```

解法二：遍历统计，统计正常人能想到的思路（还是很难）

```java
class Solution {
    public int singleNumber(int[] nums) {
        //第一个循环：统计nums数组每个元素的每一位的二进制的和
        //这个结果保存在count数组中
        int[] count = new int[32];
        for (int num : nums) {
            for (int i = 0; i < 32; i++) {
                count[i] += num & 1;//把num中第i位的二进制位累加到count[i]上
                num >>>= 1;//num右移然后与1相与，循环执行这个操作可以得到num二进制的每一位
            }
        }

        //第二个循环：从二进制统计结果count中还原res
        //先让res左移，使得res的最后一位为0；count中的每一个元素对3取余，然后添加到res的最后一位
        int res = 0, m = 3;
        for (int i = 0; i < 32; i++) {
            res <<= 1;
            res |= count[31 - i] % m;
        }
        return res;
    }
}
```



### [5.单词长度的最大乘积](https://leetcode-cn.com/problems/aseY1I/)

```java
class Solution {
    public int maxProduct(String[] words) {
        //字母有26个，int长度为32位，所以能通过设置一个int中指定位为1来表示是否一个单词中是否包含某个字母
        //例如：单词中包含a，那么对应int中第一位就为1；包含b，第二位就为1
        int[] charCount = new int[words.length];
        //遍历每个单词
        for (int i = 0; i < words.length; i++) {
            //遍历单词中的每个字母
            for (char c : words[i].toCharArray()) {
                //通过当前字母-'a'来求出当前字符相对于a的偏移量
                //a自身的偏移量为0，b为1，z为26，以此类推
                //求出偏移量后，把1向左移动相应的偏移量，并与charCount[i]做或运算
                charCount[i] |= 1 << (c - 'a');d
            }
        }
        int result = 0;
        //两两对比每个单词
        for (int i = 0; i < words.length; i++) {
            for (int j = i + 1; j < words.length; j++) {
                //若两个单词做与运算的结果为0，说明这两单词一定没有相同的字母
                if ((charCount[i] & charCount[j]) == 0) {
                    //求最大长度乘积
                    result = Math.max(result, words[i].length() * words[j].length());
                }
            }
        }
        return result;
    }
}
```



### [6. 排序数组中两个数字之和](https://leetcode-cn.com/problems/kLl5u1/)

解法一：双指针

```java
class Solution {
    public int[] twoSum(int[] numbers, int target) {
        int len = numbers.length;
        int i = 0, j = len - 1;
        while (i < j) {
            int sum = numbers[i] + numbers[j];
          	//1.若两数之和等于target，返回i和j
          	//2.若大于target，使j左移
          	//3.若小于target，使i右移
            if (sum == target) {
                return new int[]{i, j};
            } else if (sum > target) {
                j--;
            } else {
                i++;
            }
        }
        return new int[]{i, j};
    }
}
```

解法二：二分查找

```java
class Solution {
    public int[] twoSum(int[] numbers, int target) {
        for (int i = 0; i < numbers.length; i++) {
          	//固定当前数x，然后查找target-x的下标
            int x = numbers[i];
            int index = binarySearch(numbers, i + 1, numbers.length - 1, target - x);
	          //若找到了target-x的下标，就返回i和index
            if (index != -1) {
                return new int[]{i, index};
            }
        }
        return new int[0];
    }

  	/**
     * 在nums中查找整数target的下标，若不存在则返回-1
     */
    private int binarySearch(int[] nums, int left, int right, int target) {
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] > target) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return -1;
    }
}
```



## 数组

### [7. 数组中和为0的三个数](https://leetcode-cn.com/problems/1fGaJU/)

```java
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        //如果数组长度小于3，是不可能凑出三元组的
        if (nums == null || nums.length < 3) {
            return new ArrayList<>();
        }
        List<List<Integer>> res = new ArrayList<>();
        //对数组进行排序
        Arrays.sort(nums);

        //先固定i，然后找合为-nums[i]的两个数
        for (int i = 0; i < nums.length - 2; i++) {
            //跳过所有重复的i
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            int target = -nums[i];
            //在i+1到nums.length-1的区间内找和为target的两个数
            //因为进行了排序，所以使用双指针法而不是HashMap
            int left = i + 1;
            int right = nums.length - 1;
            while (left < right) {
                int sum = nums[left] + nums[right];
                if (sum == target) {
                    res.add(Arrays.asList(nums[i], nums[left], nums[right]));
                    //左右指针跳过所有重复的数
                    while (left < right && nums[left] == nums[++left]);
                    while (left < right && nums[right] == nums[--right]);
                } else if (sum > target) {
                    right--;
                } else {
                    left++;
                }
            }
        }

        return res;
    }
}
```



### [8. 和大于等于target的最短子数组](https://leetcode-cn.com/problems/2VG8Kg/)

```java
class Solution {
    public int minSubArrayLen(int target, int[] nums) {
        int left = 0;
        int sum = 0;
        int ans = Integer.MAX_VALUE;
        for (int right = 0; right < nums.length; right++) {
            sum += nums[right];
            //一旦满足了sum>=targer，就尽量缩小左边界，直到sum>=target不成立再扩大右边界
            while (sum >= target) {
                ans = Math.min(ans, right - left + 1);
                sum -= nums[left++];
            }
        }
        //若ans比length大说明没找到满足要求的连续子序列
        return ans > nums.length ? 0 : ans;
    }
}
```



### [9. 乘积小于k的子数组](https://leetcode-cn.com/problems/ZVAVXX/)

```java
class Solution {
    public int numSubarrayProductLessThanK(int[] nums, int k) {
        int left = 0;
        int ans = 0;
        //乘积初始化为1
        int product = 1;
        for (int right = 0; right < nums.length; right++) {
            //乘积累乘以滑动窗口右端的数
            product *= nums[right];
            //在满足乘积大于等于k的情况下，尽量缩小滑动窗口左端
            //由于left、right只向右移动不会回到之前的位置，所以不会重复统计满足要求的子数组
            while (left <= right && product >= k) {
                product /= nums[left++];
            }
            if (left <= right) {
                ans += right - left + 1;
            }
        }
        return ans;
    }
}
```



### [10. 和为k的子数组](https://leetcode-cn.com/problems/QTMn0o/)

```java
class Solution {
    public int subarraySum(int[] nums, int k) {
        //用map来记录前面的presum出现的次数
        Map<Integer, Integer> preSumFreq = new HashMap<>();
        preSumFreq.put(0, 1);
        //preSum用来保存从第一个元素到当前元素num的所有元素和
        int preSum = 0;
        //count用来保存和为k的子数组个数
        int count = 0;
        //遍历数组
        for (int num : nums) {
            //更新preSum的值
            preSum += num;
            //对于一个数num，我们想求有多少个以num结尾的子数组和为k
            //当前的元素总和为preSum（nums[0]+nums[1]+...+num），假设有一个数在nums[0]~num的子区间内
            //从这个数开始往p后一直到num的子数组之和为k，可以经过推导得出preSum(num) - preSum(k) = k，即preSum(k)=preSum(num)-k
            //由于我们把从数组首元素到当前的所有preSum及其出现次数都存在map里来，我们可以通过preSum - k得到preSum(k)出现的次数
            //这个次数也即是preSum(num) - preSum(k) = k这个式子能成立的次数，把它累加到count上
            if (preSumFreq.containsKey(preSum - k)) {
                count += preSumFreq.get(preSum - k);
            }
            //维护map
            preSumFreq.put(preSum, preSumFreq.getOrDefault(preSum, 0) + 1);
        }
        return count;
    }
}
```



### [11. 0和1个数相同的子数组](https://leetcode-cn.com/problems/A1NYOS/)

```java
class Solution {
    public int findMaxLength(int[] nums) {
        //map存放的是"某个前缀和出现的最小下标"
        Map<Integer, Integer> map = new HashMap<>();
        //初始化前缀和0的最小下标为-1
        //在遍历过程中若preSum为0，那从第一个元素到当前元素的长度就为当前符合条件的子数组的长度
        //如：遍历到下标为3的元素时发现preSum为0，那么当前满足条件子数组的长度就是3-(-1)=4
        map.put(0, -1);
        //preSum记录从头遍历到当前数的总前缀和
        int preSum = 0;
        //ans记录相同数量0和1的最长连续子数组的长度
        int ans = 0;
        for (int i = 0; i < nums.length; i++) {
            //若当前数为0就加上-1，这样当出现相同数量的0和1时，preSum就为0
            //反过来也可以说preSum为0说明数组存在相同数量的0和1
            preSum += nums[i] == 0 ? -1 : 1;
            //可以把map中已经存在的preSum作为preSum1,当前的总preSum作为preSum2
            //由于preSum1等于preSum2，所以从preSum1出现的下标的后一位数到当前数的子数组和就为0
            //此时就可以尝试去更新ans，若大于ans就更新为新的符合条件的子数组的长度
            if (map.containsKey(preSum)) {
                ans = Math.max(ans, i - map.get(preSum));
            } else {
                //若不存在相同的preSum，就把这个新的preSum存在map中
                map.put(preSum, i);
            }
        }
        return ans;
    }
}
```



### [12. 左右两边子数组的和相等](https://leetcode-cn.com/problems/tvdfij/)

```java
class Solution {
    public int pivotIndex(int[] nums) {
        //先计算出数组的和
        int total = 0;
        for (int num : nums) {
            total += num;
        }
        //curSum用与存储遍历过程中，从第一个元素到当前元素的所有数的和
        int curSum = 0;
        for (int i = 0; i < nums.length; i++) {
            curSum += nums[i];
            //curSum - nums[i]为当前数左边的数的和
            //total - curSum为当前数右边的数的和
            if (curSum - nums[i] == total - curSum) {
                return i;
            }
        }
        return -1;
    }
}
```



### [13. 二维子矩阵的和](https://leetcode-cn.com/problems/O4NDxx/)

```java
class NumMatrix {

    //声明一个sums数组，行列数分别为m+1，n+1
    //sums[i + 1][j + 1]表示矩阵从左上角到matrix[i][j]的子矩阵的元素和
    int[][] sums;

    public NumMatrix(int[][] matrix) {
        int m = matrix.length;
        if (m > 0) {
            int n = matrix[0].length;
            sums = new int[m + 1][n + 1];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    //当前的子矩阵总和为左侧和上面的子矩阵总和相加-重复的部分+当前元素
                    //可以看出sum[i][j]被重复加了一次
                    sums[i + 1][j + 1] = sums[i][j + 1] + sums[i + 1][j] - sums[i][j] + matrix[i][j];
                }
            }
        }
    }
    
    public int sumRegion(int row1, int col1, int row2, int col2) {
        //把要减去的，要重新加上的部分做加减就行了
        return sums[row2 + 1][col2 + 1] - sums[row1][col2 + 1] - sums[row2 + 1][col1] + sums[row1][col1];
    }
}

/**
 * Your NumMatrix object will be instantiated and called as such:
 * NumMatrix obj = new NumMatrix(matrix);
 * int param_1 = obj.sumRegion(row1,col1,row2,col2);
 */
```



## 字符串

### [14. 字符串中的变位词](https://leetcode-cn.com/problems/MPnaiL/)

```java
class Solution {
    public boolean checkInclusion(String s1, String s2) {
        //两个长度为26的整数数组，用于表示字符出现的次数
        int[] arr1 = new int[26];
        int[] arr2 = new int[26];
        if (s1.length() > s2.length()) {
            return false;
        }

        //第一次循环
        //统计s1中每个字符出现的次数
        //同时在s2形成第一个滑动窗口，这个滑动窗口的左侧为s2第一个字符，窗口大小等于s1
        for (int i = 0; i < s1.length(); i++) {
            arr1[s1.charAt(i) - 'a']++;
            arr2[s2.charAt(i) - 'a']++;
        }

        //遍历s2，逐一对比s2中的滑动窗口与s1的字符出现次数是否相等
        //若相等就返回true，否则使窗口右移
        for (int i = s1.length(); i < s2.length(); i++) {
            if (Arrays.equals(arr1, arr2)) {
                return true;
            }
            //滑动窗口右移，所以需要使得原本窗口最左端的字符的出现次数-1
            arr2[s2.charAt(i - s1.length()) - 'a']--;
            arr2[s2.charAt(i) - 'a']++;
        }

        //判断最后一个滑动窗口是否与arr1相等
        return Arrays.equals(arr1, arr2);
    }
}
```

