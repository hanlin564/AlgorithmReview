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



### [15. 字符串中的所有变位词](https://leetcode-cn.com/problems/VabMRr/)

```java
class Solution {
    public List<Integer> findAnagrams(String s, String p) {
        //ans用于保存子串的起始索引
        List<Integer> ans = new ArrayList<>();

        //用int数组表示p串中每个字符的出现次数
        int[] need = new int[26];
        for (int i = 0; i < p.length(); i++) {
            need[p.charAt(i) - 'a']++;
        }

        //初始化滑动窗口的左端和右端
        int start = 0, end = 0;
        //用window数组表示滑动窗口内每个字符出现的次数
        int[] window = new int[26];
        while (end < s.length()) {
            //上一次循环结束前，我们使得滑动窗口右端右移，所以这里需要把右端的字符的出现次数加上
            window[s.charAt(end) - 'a']++;
            //这里有个疑惑的地方：我发现while和if是可以相互替换的，最终都能ac；但是while的速度更快
            //理论上来说这里用while的话是不会进入第二轮循环的，所以用while的意义不明
            //若当前的滑动窗口字符个数等于p的字符数，那么当窗口左端右移后，右端不变，while的条件就不成立了
            while (end - start + 1 == p.length()) {
                if (Arrays.equals(window, need)) {
                    ans.add(start);
                }
                window[s.charAt(start) - 'a']--;
                start++;
            }
            end++;
        }

        return ans;
    }
}
```



### [16. 不含重复字符的最长子字符串](https://leetcode-cn.com/problems/wtcaE1/)

```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        //用map记录每个字符最后出现的下标
        Map<Character, Integer> lastIndexForChar = new HashMap<>();
        //分别声明最长连续子字符串的长度和当前子字符串的长度（皆不含重复字符）
        int maxSubLen = 0, curSubLen = 0;

        for (int i = 0; i < s.length(); i++) {
            //取出当前字符上一次出现的下标，若没有就默认为-1
            int lastIndex = lastIndexForChar.getOrDefault(s.charAt(i), -1);
            //更新当前字符的最后出现位置
            lastIndexForChar.put(s.charAt(i), i);

            //如果当前子字符串的长度小于(当前下标-上一次出现的下标)，说明子串中没有出现重复字符，可以使当前长度+1
            //若大于，则说明出现了重复字符，那么当前子串的最大长度就是i - lastIndex
            //此外当lastIndex为-1，也即是这个字符第一次出现时，i - lastIndex会使当前子串长度+1（这就是默认字符出现位置为-1的原因）
            curSubLen = curSubLen < i - lastIndex ? curSubLen + 1 : i - lastIndex;
            //尝试更新maxSubLen
            maxSubLen = Math.max(maxSubLen, curSubLen);
        }

        return maxSubLen;
    }
}
```



### [17. 含有所有字符的最短字符串](https://leetcode-cn.com/problems/M1oyTv/)

```java
class Solution {
    public String minWindow(String s, String t) {
        //排除一些非法情况
        if (s == null || s.length() == 0 || t == null || t.length() == 0) {
            return "";
        }
        //need数组用于表示每个字符的需要个数
      	//ASCII码一共有128个
        int[] need = new int[128];
        //先遍历t，记录下需要的字符个数
        for (int i = 0; i < t.length(); i++) {
            need[t.charAt(i)]++;
        }
        //left和right分别是当前滑动窗口的左右边界，size是当前滑动窗口的大小
        //count是当前需求的字符个数
        //start是最小覆盖串开始处的下标
        int left = 0, right = 0, size = Integer.MAX_VALUE, count = t.length(), start = 0;
        //遍历s字符串
        while (right < s.length()) {
            //取出当前右边界的字符
            char c = s.charAt(right);
            //need[c]大于0，说明这个字符c在t里面出现了need[c]次
            //need[c]小于等于0，说明这个字符c在t里面没有出现
            //当c在t中出现了，说明c可以放入滑动窗口内，并为"凑成覆盖字串"的目标做贡献，我们把count--以表示需要凑的字符数量减1
            if (need[c] > 0) {
                count--;
            }
            //无论c是否能为最小覆盖字串做贡献，都要对其进行need[c]--操作
            need[c]--;
            //当count为0时，说明这个滑动窗口内已经包含了全部t中的字符
            //这时就要试图把left右移来得到最小的滑动窗口
            if (count == 0) {
                //当need[s.charAt(left)] < 0时，说明左边界处的字符没对最小覆盖字串没贡献，也即是说，我们不需要s.charAt(left)
                //忽略掉这样不需要的字符以得到更小的窗口（即把左边界右移，同时更新need数组）
                while (left < right && need[s.charAt(left)] < 0) {
                    need[s.charAt(left)]++;
                    left++;
                }
                //若当前滑动窗口大小小于此前的最小滑动窗口大小，则更新size和start
                if (right - left + 1 < size) {
                    size = right - left + 1;
                    start = left;
                }
                //这里讨论下为啥要对need[s.charAt(left)]++和left++
                //上面的need[c]--操作，这导致了s中所有字符的需要次数都比实际的少1
                //那么这时的最小滑动窗口的左边界的need值是0，这是因为前面while循环的退出条件是need[s.charAt(left)] < 0
                //可是左边界的字符的需要次数为1，不然它不可能算在滑动窗口内，所以要对其进行++操作
                need[s.charAt(left)]++;
                //左边界右移，这样滑动窗口内肯定不包含t中的所有字符了，一切都要重新计算了
                left++;
                //由于左边界只右移了一位，说明只有一个字符不被包含
                //count++，表示要重新开始凑的字符由0变为1
                count++;
            }
            //右边界右移
            right++;
        }
        //在s中截取start到start+size的字串作为结果返回
        return size == Integer.MAX_VALUE ? "" : s.substring(start, start + size);
    }
}
```



### [18. 有效的回文](https://leetcode-cn.com/problems/XltzEq/)

```java
class Solution {
    public boolean isPalindrome(String s) {
        //""是回文串
        if ("".equals(s)) {
            return true;
        }

        //定义两个指针指向串的开始和结尾
        int left = 0, right = s.length() - 1;
        while (left < right) {
            //若左右两个指针指向的都不是数组或字母，则需要把它们往中间移动，直到它们都指向字母和数字为止
            if (!isDigitOrLetter(s.charAt(left))) {
                left++;
            } else if (!isDigitOrLetter(s.charAt(right))) {
                right--;
            } else {
                //统一把两个指针指向的字符转为小写再进行比较，若不相同则说明不是回文串，返回false
                if (Character.toLowerCase(s.charAt(left)) != Character.toLowerCase(s.charAt(right))) {
                    return false;
                }
                //若字符相同则使左右指针往中间移动
                left++;
                right--;
            }
        }

        //若循环中没返回false，最终默认返回true，说明这个串是回文串
        return true;
    }

    //判断一个字符是否为数字或字母
    private boolean isDigitOrLetter(char c) {
        if (c >= '0' && c <= '9' || c >= 'A' && c <= 'Z' || c >= 'a' && c <= 'z') {
            return true;
        }
        return false;
    }
}
```



### [19. 最多删除一个字符得到回文](https://leetcode-cn.com/problems/RQku0D/)

```java
class Solution {
    public boolean validPalindrome(String s) {
        if ("".equals(s)) {
            return true;
        }

        //略过开头结尾所有的回文的部分
        int left = 0, right = s.length() - 1;
        while (s.charAt(left) == s.charAt(right) && left < right) {
            left++;
            right--;
        }

        /**
         * 上面循环退出有两种情况
         * 第一种情况下退出，说明左右指针指向的字符不相等了；这时跳过左指针或右指针，去判断剩余的区间是否为回文串
         * 第二种情况下退出，说明左右指针指向了同一字符，s自身就是回文串，不需要删除字符
         */
        return left == s.length() / 2 || isPalindrome(s, left + 1, right) || isPalindrome(s, left, right - 1);
    }

    //判断s串从left到right的子串是否为回文串
    private boolean isPalindrome(String s, int left, int right) {
        while (left < right) {
            if (s.charAt(left) != s.charAt(right)) {
                return false;
            }
            left++;
            right--;
        }
        return true;
    }
}
```



### [20. 回文子字符串的个数](https://leetcode-cn.com/problems/a7VOhD/)

```java
class Solution {
    int ans=0;

    public int countSubstrings(String s) {
        for (int i = 0; i < s.length(); i++) {
            //回文子串长度为奇数(以一个字符为中心)
            extendSubstrings(s, i, i);
            //回文子串长度为偶数(以两个字符为中心)
            extendSubstrings(s, i, i+1);
        }
        return ans;
    }

    //中心扩展
    public void extendSubstrings(String s,int start,int end){
        //使ans++,且使回文子串前后都增加一个字符的条件:
        //1.回文子串的开始和结束位置在s字符串内
        //2.前后的字符是相同的
        while (start>=0 && end<s.length() && s.charAt(start)==s.charAt(end)) {
            start--;
            end++;
            ans++;
        }
    }
}
```



## 链表

### [21. 删除链表的倒数第N个节点](https://leetcode-cn.com/problems/SLwz0R/)

```java
class Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode fast = head, slow = head;
        for (int i = 0; i < n; i++) {
            fast = fast.next;
        }

        while (fast != null && fast.next != null) {
            fast = fast.next;
            slow = slow.next;
        }

        if (fast == null) {
            return head.next;
        } else {
            slow.next = slow.next.next;
            return head;
        }
    }
}
```



### [22. 链表中环的入口节点](https://leetcode-cn.com/problems/c32eOV/)

```java
/**
 * Definition for singly-linked list.
 * class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
public class Solution {
    public ListNode detectCycle(ListNode head) {
        //把链表分为环部分（b）和非环部分（a）,a与b指对应部分的链表长度（节点数）
        //快慢指针
        ListNode fast = head, slow = head;
        while (true) {
            if (fast == null || fast.next == null) {
                return null;
            }
            //fast移动两位，slow移动一位
            fast = fast.next.next;
            slow = slow.next;
            //当slow和fast第一次相遇时，肯定在环里面
            //此时slow走了n*b步（？）
          	//快指针走的步数f=2*慢指针走的步数s（f=2*s）
          	//f=s+n*b，快指针和慢指针在环中相遇，说明快指针比慢指针多走环长度的整数倍
          	//2*s=s+n*b得出s=n*b
            if (fast == slow) {
                break;
            }
        }
        //容易知道当slow走了a+nb步时都是在入环的第一个节点处
        //那么让fast重新指向head，slow和fast同时一格格移动
        //当两者再次相遇时，fast走了a步（从head到入环处），slow也走了nb+a步
        //相遇处就是入环处
        fast = head;
        while (slow != fast) {
            slow = slow.next;
            fast = fast.next;
        }
        return fast;
    }
}
```



### [23. 两个链表的第一个重合节点](https://leetcode-cn.com/problems/3u1WK4/)

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode a = headA, b = headB;
        while (a != b) {
            a = a == null ? headB : a.next;
            b = b == null ? headA : b.next;
        }
        return a;
    }
}
```



### [24. 反转链表](https://leetcode-cn.com/problems/UHnkqh/)

迭代法：

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode reverseList(ListNode head) {
        ListNode cur = head;
        ListNode ans = null;
        while (cur != null) {
            ListNode nextTmp = cur.next;
            cur.next = ans;
            ans = cur;
            cur = nextTmp;
        }
        return ans;
    }
}
```

递归法：

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode reverseList(ListNode head) {
        if(head==null || head.next==null)
            return head;
        ListNode result = reverseList(head.next);
        head.next.next=head;
        head.next=null;
        return result;
    }
}
```



### [25. 链表中的两数相加](https://leetcode-cn.com/problems/lMSNwu/)

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
      //分别为两个链表创建栈
        Stack<Integer> stack1 = buildStack(l1);
        Stack<Integer> stack2 = buildStack(l2);
      //创建一个虚的头节点
        ListNode dummy = new ListNode();
      //carry表示相加的进位，初始化为0
        int carry = 0;
      
        while (!stack1.isEmpty() || !stack2.isEmpty() || carry != 0) {
          //取得两个栈顶部的值并相加，计算进位和当前位的值
            int x = stack1.isEmpty() ? 0 : stack1.pop();
            int y = stack2.isEmpty() ? 0 : stack2.pop();
            int sum = x + y + carry;
          //当前节点值为sum对10取余，进位为整除10
            ListNode node = new ListNode(sum % 10);
            carry = sum / 10;
          //把新创建的节点插入到dummy节点之后
            node.next = dummy.next;
            dummy.next = node;
        }
      
        return dummy.next;
    }

  //根据链表去创建一个栈，从栈底到栈顶依次是链表头部的值到链表尾部的值
    private Stack<Integer> buildStack(ListNode listNode) {
        Stack<Integer> stack = new Stack<>();
        while (listNode != null) {
            stack.push(listNode.val);
            listNode = listNode.next;
        }
        return stack;
    }
}
```



### [26. 重排链表](https://leetcode-cn.com/problems/LGjMqU/)

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public void reorderList(ListNode head) {
        //反转链表的后一半，再与前一半合并
        ListNode mid = getMiddle(head);
        ListNode a = head;
        ListNode b = mid.next;
        mid.next = null;
        mergeList(a, reverseList(b));
    }

    //找到链表的中点
    private ListNode getMiddle(ListNode head) {
        ListNode slow = head;
        ListNode fast = head;
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        return slow;
    }

    //反转链表
    private ListNode reverseList(ListNode head) {
        ListNode cur = head;
        ListNode ans = null;
        while (cur != null) {
            ListNode nextTmp = cur.next;
            cur.next = ans;
            ans = cur;
            cur = nextTmp;
        }
        return ans;
    }

    //合并两个链表
    //a和b交错合并，a的头节点为合并结果的头节点
    private void mergeList(ListNode a, ListNode b) {
        ListNode aTmp, bTmp;
        while (a != null && b != null) {
            aTmp = a.next;
            bTmp = b.next;
            a.next = b;
            a = aTmp;
            b.next = a;
            b = bTmp;
        }
    }
}
```



### [27. 回文链表](https://leetcode-cn.com/problems/aMhZSa/)

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public boolean isPalindrome(ListNode head) {
        if (head == null || head.next == null) {
            return true;
        }
        ListNode slow = head, fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        //fast == null:偶数节点，slow指向两个中间节点靠后的那一个
        //fast != null:奇数节点，slow指向唯一的中间节点
        cut(head, slow);
      //先把链表前后两半拆开，再翻转后一半，最后判断节点值是否相等
        return isEqual(head, reverseList(slow));
    }

  //反转链表
    private ListNode reverseList(ListNode head) {
        ListNode res = null;
        ListNode cur = head;
        while (cur != null) {
            ListNode nextTmp = cur.next;
            cur.next = res;
            res = cur;
            cur = nextTmp;
        }
        return res;
    }

  //判断两个链表的节点值是否相等，当两个链表的节点数相差在1之内，对结果不会有影响
    private boolean isEqual(ListNode headA, ListNode headB) {
        while (headA != null && headB != null) {
            if (headA.val != headB.val) {
                return false;
            }
            headA = headA.next;
            headB = headB.next;
        }
        return true;
    }
    
  //断开以head为头节点的链表中cutNode之后（包括）的部分
    private void cut(ListNode head, ListNode cutNode) {
        while (head.next != cutNode) {
            head = head.next;
        }
        head.next = null;
    }
}
```



### [28. 展平多级双向链表](https://leetcode-cn.com/problems/Qv1Da2/)

```java
/*
// Definition for a Node.
class Node {
    public int val;
    public Node prev;
    public Node next;
    public Node child;
};
*/

class Solution {
  //抄的题解，用dfs做的，代码不难，但是我根本是题都看不懂
    List<Node> list = new ArrayList<>();

    public Node flatten(Node head) {
        dfs(head);
        for (int i = 0; i < list.size() - 1; i++) {
            Node pre = list.get(i);
            Node cur = list.get(i + 1);
            cur.prev = pre;
            pre.next = cur;
            pre.child = null;
        }
        return head;
    }

    private void dfs(Node head) {
        if (head == null) return;
        list.add(head);
        dfs(head.child);
        dfs(head.next);
    }
}
```



### [29. 排序的循环链表](https://leetcode-cn.com/problems/4ueAj6/)

```java
/*
// Definition for a Node.
class Node {
    public int val;
    public Node next;

    public Node() {}

    public Node(int _val) {
        val = _val;
    }

    public Node(int _val, Node _next) {
        val = _val;
        next = _next;
    }
};
*/

class Solution {
    public Node insert(Node head, int insertVal) {
        //当链表头为null，就以insertVal创建head节点并使head的后继指向自身（循环链表），然后返回head
        if (head == null) {
            head = new Node(insertVal);
            head.next = head;
            return head;
        }

        //遍历循环链表
        Node cur = head;
        while (cur.next != head) {
            //若后继节点的节点值小于当前节点值，说明当前节点是循环链表的尾部
            if (cur.next.val < cur.val) {
                //比最小的还小、比最大的还大这两种情况都可插入
                if (cur.next.val >= insertVal) {
                    break;
                } else if (cur.val <= insertVal) {
                    break;
                }
            }
            //在链表中间找到了插入的位置
            if (cur.val <= insertVal && cur.next.val >= insertVal) {
                break;
            }
            cur = cur.next;
        }

        //把新节点创建并插入，最后返回head
        cur.next = new Node(insertVal, cur.next);
        return head;
    }
}
```



## 哈希表

### [30. 插入、删除和随机访问都是O(1)的容器](https://leetcode-cn.com/problems/FortPu/)

```java
class RandomizedSet {

    /**
     * indexForNums存储每个数在list中的下标
     * list为动态数组，存储元素值.使用list的原因是单单map无法通过随机数来取值
     * random用于随机取值
     */
    Map<Integer, Integer> indexForNums;
    List<Integer> list;
    Random random = new Random();

    /** Initialize your data structure here. */
    public RandomizedSet() {
        indexForNums = new HashMap<>();
        list = new ArrayList<>();
    }
    
    /** Inserts a value to the set. Returns true if the set did not already contain the specified element. */
    public boolean insert(int val) {
        //若容器中已存在val，就返回false
        if (indexForNums.containsKey(val)) {
            return false;
        }

        //把val放在数组最后，并更新indexForNums中的映射
        indexForNums.put(val, list.size());
        list.add(list.size(), val);
        return true;
    }
    
    /** Removes a value from the set. Returns true if the set contained the specified element. */
    public boolean remove(int val) {
        //若容器中不存在val，就返回false
        if (!indexForNums.containsKey(val)) {
            return false;
        }

        //把val与list中最后一个数做交换，然后删除最后一个数
        int lastNum = list.get(list.size() - 1);
        int index = indexForNums.get(val);
        list.set(index, lastNum);
        indexForNums.put(lastNum, index);
        list.remove(list.size() - 1);
        indexForNums.remove(val);
        return true;
    }
    
    /** Get a random element from the set. */
    public int getRandom() {
        //因为用了数组存储值，所以可以根据随机数来随机获取容器中的值
        return list.get(random.nextInt(list.size()));
    }
}

/**
 * Your RandomizedSet object will be instantiated and called as such:
 * RandomizedSet obj = new RandomizedSet();
 * boolean param_1 = obj.insert(val);
 * boolean param_2 = obj.remove(val);
 * int param_3 = obj.getRandom();
 */
```



### [31. 最少最近使用缓存](https://leetcode-cn.com/problems/FortPu/)

```java
class LRUCache {

    class Node {
        int key;
        int value;
        Node next, prev;
        Node() {}
        Node(int key, int value) {
            this.key = key;
            this.value = value;
        }
    }

    Node head, tail;
    Map<Integer, Node> cache;
    int capacity, size;

    public LRUCache(int capacity) {
        this.capacity = capacity;
        size = 0;
        cache = new HashMap<>();
        head = new Node();
        tail = new Node();
        head.next = tail;
        tail.prev = head;
    }
    
    public int get(int key) {
        Node node = cache.get(key);
        if (node == null) {
            return -1;
        } else {
            moveToHead(node);
            return node.value;
        }
    }
    
    public void put(int key, int value) {
        Node node = cache.get(key);
        if (node == null) {
            node = new Node(key, value);
            cache.put(key, node);
            addToHead(node);
            size++;
            if (size > capacity) {
                Node tail = removeTail();
                cache.remove(tail.key);
                size--;
            }
        } else {
            node.value = value;
            moveToHead(node);
        }
    }

    private void moveToHead(Node node) {
        removeNode(node);
        addToHead(node);
    }

    private void addToHead(Node node) {
        node.prev = head;
        node.next = head.next;
        head.next = node;
        node.next.prev = node;
    }

    private void removeNode(Node node) {
        node.prev.next = node.next;
        node.next.prev = node.prev;
    }

    private Node removeTail() {
        Node oldTail = tail.prev;
        removeNode(oldTail);
        return oldTail;
    }
}

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache obj = new LRUCache(capacity);
 * int param_1 = obj.get(key);
 * obj.put(key,value);
 */
```



### [32. 有效的变位词](https://leetcode-cn.com/problems/dKk3P7/)

```java
class Solution {
    public boolean isAnagram(String s, String t) {
        //若两字符串长度不等，则一定不是互为变位词
        //若两字符串的相等，说明它们的字符顺序完全相同，不是变位词
        if (s.length() != t.length() || s.equals(t)) {
            return false;
        }

        //使用counts数组统计每个小写字母出现的次数
        int[] counts = new int[26];

        //先统计s的字符出现次数，再对t中出现字符做减法
        for (char c : s.toCharArray()) {
            counts[c - 'a']++;
        }
        for (char c : t.toCharArray()) {
            counts[c - 'a']--;
        }
        //减去t中字符出现次数后，最终若有字符出现次数不为0，说明两个字符串有同个字符出现不同次数的情况，不是变位词
        for (int count : counts) {
            if (count != 0) {
                return false;
            }
        }

        return true;
    }
}
```



### [33. 变位词组](https://leetcode-cn.com/problems/sfvd7V/)

```java
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        //同变位词的字符排序后生成的string -> 存储同变位词的数组
        Map<String, ArrayList<String>> map = new HashMap<>();
        for (String str : strs) {
            //把当前字符串转为字符数组并排序，再用排序后的字符数组去创建一个名为key的字符串
            char[] array = str.toCharArray();
            Arrays.sort(array);
            String key = new String(array);
            //所有当前字符的变位词的key都是一样的，所以可以通过key来定位map中存储了所有当前字符变位词的数组
            //若没有目标数组就new一个
            ArrayList<String> list = map.getOrDefault(key, new ArrayList<>());
            //把当前字符串加入到存储它的变位词的数组中
            list.add(str);
            map.put(key, list);
        }
        //把map的所有值都转化为题目要求的嵌套数组
        return new ArrayList<List<String>>(map.values());
    }
}
```



### [34. 外星语言是否排序](https://leetcode-cn.com/problems/lwyVBB/)

```java
class Solution {
    public boolean isAlienSorted(String[] words, String order) {
        //orderIndex数组用于统计26个小写字母在外星语中的字母顺序
        //比如a在外星语中的顺序是2，则orderIndex[0]=2
        int[] orderIndex = new int[26];
        for (int i = 0; i < 26; i++) {
            orderIndex[order.charAt(i) - 'a'] = i;
        }

        //遍历单词序列，两两对比单词
        for (int i = 0; i < words.length - 1; i++) {
            char[] word1 = words[i].toCharArray();
            char[] word2 = words[i + 1].toCharArray();
            //同时开始遍历两个单词，判断两个单词同一个位置的字母的字典序大小
            for (int j = 0; j < Math.max(word1.length, word2.length); j++) {
                //若其中一个单词已经遍历到头了（j >= word.length），就把对应的index置为-1，代表这个单词的顺序一定不会大于另一个
                //若没有遍历到头，就从orderIndex中取出当前字母在外星语中对应的顺序
                int index1 = j >= word1.length ? -1 : orderIndex[word1[j] - 'a'];
                int index2 = j >= word2.length ? -1 : orderIndex[word2[j] - 'a'];
                //若index1小于index2，说明两个单词当前指向字母是符合字典序的，退出循环，进行下一轮对比
                //否则说明是不符合字典序的，直接返回false
                //若相等就继续进行循环，进行下一个字母的对比
                if (index1 < index2) {
                    break;
                }
                if (index1 > index2) {
                    return false;
                }
            }
        }

        return true;
    }
}
```



### [35. 最小时间差](https://leetcode-cn.com/problems/569nqc/)

```java
class Solution {
    public int findMinDifference(List<String> timePoints) {
        //如果时间点总数超过24*60说明肯定有重复，返回0
        if (timePoints.size() > 24 * 60) {
            return 0;
        }

        //把每个时间点都转换为分钟的形式，加入到数组中，并从小到大排序
        List<Integer> minutes = new ArrayList<>();
        for (String timePoint : timePoints) {
            String[] time = timePoint.split(":");
            minutes.add(Integer.parseInt(time[0]) * 60 + Integer.parseInt(time[1]));
        }
        Collections.sort(minutes);

        //把最小值加上24*60以处理答案是最大和最小值的差值的情况
        minutes.add(minutes.get(0) + 24 * 60);

        //逐一对比两个相邻时间的差并尝试更新最小差值
        int res = 24 * 60;
        for (int i = 1; i < minutes.size(); i++) {
            res = Math.min(res, minutes.get(i) - minutes.get(i - 1));
        }

        return res;
    }
}
```



## 栈

### [36. 后缀表达式](https://leetcode-cn.com/problem-list/e8X3pBZi/)

````java
class Solution {
    public int evalRPN(String[] tokens) {
        //用栈来存储这些数
        Stack<Integer> stack = new Stack<>();
        //遍历tokens数组
        for (String token : tokens) {
            //判断当前字符是数字还是运算符
            if ("+".equals(token) || "-".equals(token) || "*".equals(token) || "/".equals(token)) {
                //是运算符号，就把栈顶两个数弹出来做相应的运算并把结果入栈
                int num2 = stack.pop();
                int num1 = stack.pop();
                switch (token) {
                    case "+": {
                        stack.push(num1 + num2);
                        break;
                    }
                    case "-": {
                        stack.push(num1 - num2);
                        break;
                    }
                    case "*": {
                        stack.push(num1 * num2);
                        break;
                    }
                    case "/": {
                        stack.push(num1 / num2);
                        break;
                    }
                }
            } else {
                //是数字，就把数字压入栈
                stack.push(Integer.valueOf(token));
            }
        }
        //栈顶作为结果返回，此时栈中也只有一个数了
        return stack.peek();
    }
}
````



### [37. 小行星碰撞](https://leetcode-cn.com/problems/XagZNi/)

```java
class Solution {
    public int[] asteroidCollision(int[] asteroids) {
        //栈用于存放从左到右遍历过程中没爆炸的小行星
        Stack<Integer> stack = new Stack<>();

        //index用于控制asteroids数组的遍历
        int index = 0;
        while (index < asteroids.length) {
            //把当前数入栈的条件有三
            //1：栈为空
            //2：栈顶小于0。若当前数小于0，一起向左移动；若当前数大于0，反向移动
            //3：当前数大于0。若栈顶小于0，反向移动；若栈顶大于0，一起向右移动
            if (stack.empty() || stack.peek() < 0 || asteroids[index] > 0) {
                stack.push(asteroids[index]);
            } else if (stack.peek() <= -asteroids[index]) {
                //若栈顶等于当前数的绝对值，且栈顶大于0，当前数小于0，两者会一起爆炸
                //这种情况就让栈顶出栈，并让index跳过当前数

                //若栈顶小于当前数的绝对值，且栈顶大于0，当前数小于0，栈顶元素会爆炸
                //当前数继续与下一个栈顶做对比（index不变），跳过这次循环
                if (stack.pop() < -asteroids[index]) {
                    continue;
                }
            }
            //当前数被撞毁，直接来到这里了，上面的else if块是不会进入的
            index++;
        }

        //把栈中的数放入数组中返回
        int[] ans = new int[stack.size()];
        for (int i = ans.length - 1; i >= 0; i--) {
            ans[i] = stack.pop();
        }
        return ans;
    }
}
```



### [38.每日温度](https://leetcode-cn.com/problems/iIQa4I/)

```java
class Solution {
    public int[] dailyTemperatures(int[] temperatures) {
        int n = temperatures.length;
        //存储答案的数组
        int[] ans = new int[n];
        //栈:用于存储原数组中的下标
        //顺序:从栈底到栈顶升序
        //如果一个下标在单调栈里，则表示尚未找到下一次温度更高的下标
        Stack<Integer> indexs = new Stack<>();
        //顺序遍历数组
        for (int curIndex = 0; curIndex < n; curIndex++) {
            //如果当前元素比栈顶中的元素要大
            //再次遍历,计算当前元素和栈顶元素中间每个下标应该保存的天数
            //循环条件:当前元素大于栈顶元素且栈不为空
            while (!indexs.isEmpty() && temperatures[curIndex] > temperatures[indexs.peek()]) {
                //栈中可能保存了很多小于当前温度的下标(栈底到栈顶从低到高)
                //通过遍历(下标从后往前)把这些下标逐一出栈,同时更新preIndex的下标(逐渐变小)
                int preIndex = indexs.pop();
                //ans[preIndex]逐渐变大
                ans[preIndex] = curIndex - preIndex;
            }
            //把数组下标入栈
            indexs.add(curIndex);
        }
        return ans;
    }
}
```



### [39. 直方图最大矩形面积](https://leetcode-cn.com/problems/0ynMMM/)

https://leetcode-cn.com/problems/largest-rectangle-in-histogram/solution/bao-li-jie-fa-zhan-by-liweiwei1419/

```java
class Solution {
    public int largestRectangleArea(int[] heights) {
        int len = heights.length;
        if (len == 1) {
            return heights[0];
        }

        //下面这些扩大数组并把首末元素都赋值为0的操作是为了方便计算
        //想象柱状图的左右两侧各有一个高度为0的柱子
        //这样就算是最低的柱子也能通过下面的循环计算出最大面积
        int res = 0;
        int[] newHeights = new int[len + 2];
        newHeights[0] = 0;
        System.arraycopy(heights, 0, newHeights, 1, len);
        newHeights[len + 1] = 0;
        len += 2;
        heights = newHeights;

        //这个栈存放的是数组从左到右的下标，且栈底到栈顶保持不严格递增（单调栈）
        //这里说下为什么要用ArrayDeque而不直接用Stack
        //因为Stack默认是动态数组Vector实现的，会有扩容和加锁导致速度下降
        //而ArrayDeque可以指定初始化多大的数组
        Deque<Integer> deque = new ArrayDeque<>(len);
        deque.addLast(0);

        for (int i = 1; i < len; i++) {
            //由于这是单调栈，所以栈顶下面的柱长肯定不大于栈顶；而heights[i]又小于栈顶，
            //所以可以确定heights[stack.peekLast()]对应的最大面积的矩形
            //若当前元素小于栈顶下标对应的元素，我们就可以确定从栈顶下标到当前下标的矩形面积
            while (heights[i] < heights[deque.peekLast()]) {
                int curHeight = heights[deque.pollLast()];
                int curWidth = i - deque.peekLast() - 1;//由于矩形宽度不包括i，所以要-1
                res = Math.max(res, curHeight * curWidth); //与res做对比，取大的那个
            }
            deque.addLast(i);
        }

        return res;
    }
}
```



### [40. 矩阵中最大的矩形](https://leetcode-cn.com/problems/PLYXKQ/)

```java
class Solution {
    public int maximalRectangle(String[] matrix) {
        if (matrix.length == 0) {
            return 0;
        }

        //这里的height是从上往下算的，且它代表之前遍历的行到当前行连续'1'的个数
        //'0'就是heights中的空气部分，'1'就是heights中的柱形部分
        //维护heights，把值传给largestRectangleArea计算就行
        int[] heights = new int[matrix[0].length()];
        //保存临时的最大矩形面积
        int maxArea = 0;

        for (int row = 0; row < matrix.length; row++) {
            for (int col = 0; col < matrix[0].length(); col++) {
                //若出现'0'，就把height重置为0
                //计算连续'1'的个数
                if (matrix[row].charAt(col) == '1') {
                    heights[col] += 1;
                } else {
                    heights[col] = 0;
                }
            }
            maxArea = Math.max(maxArea, largestRectangleArea(heights));
        }

        return maxArea;
    }

    private int largestRectangleArea(int[] heights) {
        int len = heights.length;
        if (len == 1) {
            return heights[0];
        }

        //下面这些扩大数组并把首末元素都赋值为0的操作是为了方便计算
        //想象柱状图的左右两侧各有一个高度为0的柱子
        //这样就算是最低的柱子也能通过下面的循环计算出最大面积
        int res = 0;
        int[] newHeights = new int[len + 2];
        newHeights[0] = 0;
        System.arraycopy(heights, 0, newHeights, 1, len);
        newHeights[len + 1] = 0;
        len += 2;
        heights = newHeights;

        //这个栈存放的是数组从左到右的下标，且栈底到栈顶保持不严格递增（单调栈）
        //这里说下为什么要用ArrayDeque而不直接用Stack
        //因为Stack默认是动态数组Vector实现的，会有扩容和加锁导致速度下降
        //而ArrayDeque可以指定初始化多大的数组
        Deque<Integer> deque = new ArrayDeque<>(len);
        deque.addLast(0);

        for (int i = 1; i < len; i++) {
            //由于这是单调栈，所以栈顶下面的柱长肯定不大于栈顶；而heights[i]又小于栈顶，
            //所以可以确定heights[stack.peekLast()]对应的最大面积的矩形
            //若当前元素小于栈顶下标对应的元素，我们就可以确定从栈顶下标到当前下标的矩形面积
            while (heights[i] < heights[deque.peekLast()]) {
                int curHeight = heights[deque.pollLast()];
                int curWidth = i - deque.peekLast() - 1;//由于矩形宽度不包括i，所以要-1
                res = Math.max(res, curHeight * curWidth); //与res做对比，取大的那个
            }
            deque.addLast(i);
        }

        return res;
    }
}
```



## 队列

### [41. 滑动窗口的平均值](https://leetcode-cn.com/problems/qIsx9U/)

```java
class MovingAverage {

    /**
     * length：滑动窗口大小
     * queue：存储滑动窗口中的元素的队列，队头存放滑动窗口的最左侧元素，队尾存放滑动窗口的最右侧元素
     * sum：存放滑动窗口中元素的和
     */
    private int length;
    private Queue<Integer> queue;
    private double sum = 0;

    /** Initialize your data structure here. */
    public MovingAverage(int size) {
        length = size;
        queue = new LinkedList<>();
        sum = 0;
    }
    
    public double next(int val) {
        //先检查滑动窗口中元素个数是否等于窗口大小，若相等就让队头出队，并让sum减去这个出队的值
        if (queue.size() == length) {
            sum -= queue.poll();
        }
        //后续加入的元素入队，并使sum加上这个新的值
        queue.add(val);
        sum += val;
        //计算此时滑动窗口的平均值，注意是除以队列的长度而不是滑动窗口大小
        return sum / queue.size();
    }
}

/**
 * Your MovingAverage object will be instantiated and called as such:
 * MovingAverage obj = new MovingAverage(size);
 * double param_1 = obj.next(val);
 */
```



### [42. 最近请求次数](https://leetcode-cn.com/problems/H8086Q/)

```java
class RecentCounter {

    /**
     * queue：用于存储请求的队列
     * 队头存储较早的请求，队尾存储最新的请求
     */
    private Queue<Integer> queue;

    public RecentCounter() {
        queue = new LinkedList<>();
    }
    
    public int ping(int t) {
        //最新的请求入队
        queue.add(t);
        //通过while循环使最新请求3000毫秒之前的请求都出队
        while (queue.peek() < t - 3000) {
            queue.poll();
        }
        //此时队列中的元素个数就是[t-3000, t]区间的请求个数
        return queue.size();
    }
}

/**
 * Your RecentCounter object will be instantiated and called as such:
 * RecentCounter obj = new RecentCounter();
 * int param_1 = obj.ping(t);
 */
```



### [43. 往完全二叉树添加节点](https://leetcode-cn.com/problems/NaqhDT/)

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class CBTInserter {

    List<TreeNode> list;

    public CBTInserter(TreeNode root) {
        //按层序遍历把给定完全二叉树的节点存在list中
        list = new ArrayList<>();
        Queue<TreeNode> queue = new ArrayDeque<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            list.add(node);
            if (node.left != null) {
                queue.offer(node.left);
            }
            if (node.right != null) {
                queue.offer(node.right);
            }
        }
    }
    
    public int insert(int v) {
        //用传入的值创建一个新的节点并插入到数组之后
        TreeNode node = new TreeNode(v);
        list.add(node);
        //通过list.size() / 2 - 1得到父节点的编号
        //父节点编号=子节点编号/2，但是这里的list.size()比子节点编号大1，所以要减1
        int parentIndex = list.size() / 2 - 1;
        //判断下是插入到左子树还是右子树
        if (list.size() % 2 == 0) {
            list.get(parentIndex).left = node;
        } else {
            list.get(parentIndex).right = node;
        }
        return list.get(parentIndex).val;
    }
    
    public TreeNode get_root() {
        return list.get(0);
    }
}

/**
 * Your CBTInserter object will be instantiated and called as such:
 * CBTInserter obj = new CBTInserter(root);
 * int param_1 = obj.insert(v);
 * TreeNode param_2 = obj.get_root();
 */
```



### [44. 二叉树每层的最大值](https://leetcode-cn.com/problems/hPov7L/)

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public List<Integer> largestValues(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();

        if (root == null) {
            return res;
        }
        queue.offer(root);

        //层序遍历
        //记录每一层的最大值，初始值为这层最左侧的值，并与这一层的其它节点做比对
        //求出这一层的最大值，然后插入到数组中
        while (!queue.isEmpty()) {
            int max = queue.peek().val;

            for (int i = queue.size(); i > 0; i--) {
                TreeNode node = queue.poll();
                max = max > node.val ? max : node.val;
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            }

            res.add(max);
        }

        return res;
    }
}
```



### [45. 二叉树最底层最左边的值](https://leetcode-cn.com/problems/LwUNpT/)

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public int findBottomLeftValue(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);

        /**
         * 依然是层序遍历，不过是反向层序遍历，也就是每层是从右到左的顺序
         * 等层序遍历结束，root指向的就是最底层最左侧的节点
         */
        while (!queue.isEmpty()) {
            root = queue.poll();
            if (root.right != null) {
                queue.offer(root.right);
            }
            if (root.left != null) {
                queue.offer(root.left);
            }
        }

        return root.val;
    }
}
```


### [46. 二叉树的右侧视图](https://leetcode-cn.com/problems/WNC0Lk/)

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public List<Integer> rightSideView(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        if (root != null) {
            queue.add(root);
        }
        List<Integer> res = new LinkedList<>();

        //层序遍历
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode node = queue.poll();
                //判断下当前节点是否为这一层的最后一个节点，是的话就加入到res中
                if (i == size - 1) {
                    res.add(node.val);
                }
                if (node.left != null) {
                    queue.add(node.left);
                }
                if (node.right != null) {
                    queue.add(node.right);
                }
            }
        }

        return res;
    }
}
```



## 树

### [47. 二叉树剪枝](https://leetcode-cn.com/problems/pOCWxh/)

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public TreeNode pruneTree(TreeNode root) {
        if (root == null) {
            return root;
        }

        //递归地对左右子树做处理
        root.left = pruneTree(root.left);
        root.right = pruneTree(root.right);
        //若当前节点值为0且它的左右子数都被剪枝，那么它也要被剪枝
        if (root.val == 0 && root.left == null && root.right == null) {
            root = null;
        }

        return root;
    }
}
```



### [48. 序列化与反序列化二叉树](https://leetcode-cn.com/problems/h54YBf/)

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
public class Codec {

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        if (root == null)   return "[]";
        StringBuilder res = new StringBuilder("[");
        //使用队列进行层序遍历
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()){
            TreeNode node = queue.poll();
            //不要忘记在新添加到res中的数后面加","
            if (node != null){
                res.append(node.val + ",");
                queue.add(node.left);
                queue.add(node.right);
            }else {
                res.append("null,");
            }
        }
        //删除最后一个逗号，并换成]
        res.deleteCharAt(res.length() - 1);
        res.append("]");
        return res.toString();
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        if (data.equals("[]"))  return null;
        //忽略data中的"["和"]",然后以逗号分割成string数组，每个数组元素代表一个节点值
        String[] vals = data.substring(1, data.length() - 1).split(",");
        //把数组首元素作为根结点入队
        TreeNode root = new TreeNode(Integer.parseInt(vals[0]));
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        //i为访问vals的下标，由于root已经加到queue中，所以i从1开始
        int i = 1;
        while (!queue.isEmpty()){
            TreeNode node = queue.poll();
            //当节点值为null时，其实在二叉树中就不存在这个节点，也就不需要创建TreeNode了
            //但是无论如何，由于vals[i]被访问过，所以i++无论怎样都要自增的
            if (!vals[i].equals("null")){
                node.left = new TreeNode(Integer.parseInt(vals[i]));
                queue.add(node.left);
            }
            i++;
            if (!vals[i].equals("null")){
                node.right = new TreeNode(Integer.parseInt(vals[i]));
                queue.add(node.right);
            }
            i++;
        }
        return root;
    }
}

// Your Codec object will be instantiated and called as such:
// Codec ser = new Codec();
// Codec deser = new Codec();
// TreeNode ans = deser.deserialize(ser.serialize(root));
```


### [49. 从根节点到叶节点的路径数字之和](https://leetcode-cn.com/problems/3Etpl5/)

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public int sumNumbers(TreeNode root) {
        return dfs(root, 0);
    }

    //深度优先搜索
    private int dfs(TreeNode root, int preSum) {
        if (root == null) {
            return 0;
        }

        //从上至下进行递归
        //先计算出树的上面的和，再通过乘10的方式累加上下面的节点值
        int sum = 10 * preSum + root.val;
        //是叶子结点，说明递归到头了，直接把从跟节点到当前节点的和返回
        if (root.left == null && root.right == null) {
            return sum;
        } else {
            //不是叶子结点，那么就递归地求左右子树的和
            return dfs(root.left, sum) + dfs(root.right, sum);
        }
    }
}
```


### [50. 向下的路径节点之和](https://leetcode-cn.com/problems/6eUYwP/)

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    //求有多少条路径上的节点值的和为sum
    public int pathSum(TreeNode root, int sum) {
        //若树为null,返回0
        if(root == null)    return 0;
        //满足条件的路径总数=根节点就满足条件的路径数+左子树满足条件的路径数+右子树满足条件的路径数
        int result = pathSumStartWithRoot(root, sum)+pathSum(root.left,sum)+pathSum(root.right,sum);
        return result;
    }

    public int pathSumStartWithRoot(TreeNode root, int sum){
        //如果节点为null,返回0
        if(root == null)    return 0;
        //默认路径和为0
        int pathsum = 0;
        //如果节点值就为sum,使路路径和递增
        if(root.val == sum) pathsum++;
        //当前节点为根的所有路径和=左子树的路径和+右子树的路径和+之前的路径和
        //要记得左右子树上的sum要减去当前节点的值
        pathsum += pathSumStartWithRoot(root.left, sum-root.val) + pathSumStartWithRoot(root.right, sum-root.val);
        return pathsum;
    }
}
```


### [51. 节点之和最大的路径](https://leetcode-cn.com/problems/jC7MId/)

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    private int result = Integer.MIN_VALUE;

    public int maxPathSum(TreeNode root) {
        /**
        对于任意一个节点, 如果最大和路径包含该节点, 那么只可能是两种情况:
        1. 其左右子树中所构成的和路径值较大的那个加上该节点的值后向父节点回溯构成最大路径
        2. 左右子树都在最大路径中, 加上该节点的值构成了最终的最大路径
        **/
        getMax(root);
        return result;
    }

    //getMax递归本身算的是上述第一种情况的最大路径，ret算的是第二种情况的最大路径
    //返回以node为根结点的左右子树所构成的和路径值较大的那个加上该节点的值
    private int getMax(TreeNode root) {
        if (root == null) {
            return 0;
        }
        // 如果子树路径和为负则应当置0表示最大路径不包含子树
        int left = Math.max(0, getMax(root.left));
        int right = Math.max(0, getMax(root.right));
        // 判断在该节点包含左右子树的路径和是否大于当前最大路径和
        // 这里是计算左右子树路径和+根结点值，若大于则更新ret
        result = Math.max(result, root.val + left + right);

        return Math.max(left, right) + root.val;
    }
}
```


### [52. 展平二叉搜索树](https://leetcode-cn.com/problems/NYBBNL/)

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {

    //cur指向已展开部分的最右侧的节点
    private TreeNode cur;

    public TreeNode increasingBST(TreeNode root) {
        TreeNode dummy = new TreeNode();
        cur = dummy;
        inorder(root);
        return dummy.right;
    }

    //中序遍历二叉搜索树，得到从小到大的序列
    private void inorder(TreeNode node) {
        if (node == null) {
            return;
        }

        inorder(node.left);
        //把当前节点接在cur的右子树，并更新cur的指向
        cur.right = node;
        node.left = null;
        cur = node;

        inorder(node.right);    
    }
}
```


### [53. 二叉搜索树中的中序后继](https://leetcode-cn.com/problems/P5rCT8/)

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    //标记p节点是否被访问到
    private boolean hasVisited;

    public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
        hasVisited = false;
        return inorder(root, p);
    }

    //中序遍历二叉搜索树，序列从小到大
    public TreeNode inorder(TreeNode node, TreeNode p) {
        if (node == null) {
            return null;
        }
        //若当前节点就是p，则把hasVisited置为true
        if (node == p) {
            hasVisited = true;
        }
        TreeNode left = inorder(node.left, p);
        //用if第一次判断p已经被访问到且当前节点值大于p的节点值，说明当前节点就是p的中序后继
        //把hasVisited置为true且返回当前节点作为答案
        //之所以把hasVisited置为true，是因为后面的节点都不是直接后继了
        if (hasVisited && node.val > p.val) {
            hasVisited = false;
            return node;
        }
        TreeNode right = inorder(node.right, p);
        //p的中序后继在递归中作为结果返回，保存在left或right中
        return left == null ? right : left;
    }
}
```


### [54. 所有大于等于节点的值之和](https://leetcode-cn.com/problems/w6cpku/)

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    //sum保存了当前遍历到的节点右子树(所有大于当前节点的节点)的值的和
    int sum=0;

    public TreeNode convertBST(TreeNode root) {
        traver(root);
        return root;
    }

    //通过反向中序遍历来使sum逐渐累加
    //遍历是以"最右侧的叶子节点->根节点->左子树"的顺序进行的
    //这样每个节点都可以使自己的值变为右子树值的和
    public void traver(TreeNode node){
        if(node==null)  return;
        traver(node.right);
        sum += node.val;
        node.val=sum;
        traver(node.left);
    }
}
```


### [55. 二叉搜索树迭代器](https://leetcode-cn.com/problems/kTOapQ/)

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class BSTIterator {
    //cur指向当前节点，执行next方法时，会返回cur节点的中序后继
    private TreeNode cur;
    //stack用于中序遍历
    private Stack<TreeNode> stack;

    public BSTIterator(TreeNode root) {
        cur = root;
        stack = new Stack<>();
    }
    
    //这题考的就是怎么不用递归去实现中序遍历
    public int next() {
        //把当前节点一直到最左侧的叶子结点都入栈，循环结束后cur指向null
        while (cur != null) {
            stack.push(cur);
            cur = cur.left;
        }
        //cur指向刚出栈的节点，并把这个节点值返回
        cur = stack.pop();
        int ans = cur.val;
        //cur指向cur的右子节点
        //这样下一次执行next方法就是针对cur的右子树进行了（中序遍历）
        cur = cur.right;
        return ans;
    }
    
    public boolean hasNext() {
        return cur != null || !stack.isEmpty();
    }
}

/**
 * Your BSTIterator object will be instantiated and called as such:
 * BSTIterator obj = new BSTIterator(root);
 * int param_1 = obj.next();
 * boolean param_2 = obj.hasNext();
 */
```


### [56. 二叉搜索树中两个节点之和](https://leetcode-cn.com/problems/opLdQZ/)

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public boolean findTarget(TreeNode root, int k) {
        List<Integer> nums = new ArrayList<>();
        inorder(root, nums);
        //通过二分查找寻找是否存在和为k的两个节点
        int i = 0, j = nums.size() - 1;
        while (i < j) {
            int sum = nums.get(i) + nums.get(j);
            if (sum == k) {
                return true;
            } else if (sum > k) {
                j--;
            } else {
                i++;
            }
        }
        return false;
    }

    //通过中序遍历得到二叉搜索树从小到大的顺序序列，并存放在nums中
    private void inorder(TreeNode root, List<Integer> nums) {
        if (root == null) {
            return;
        }
        inorder(root.left, nums);
        nums.add(root.val);
        inorder(root.right, nums);
    }
}
```


### [57. 值和下标之差都在给定范围内](https://leetcode-cn.com/problems/7WqeDu/)

```java
class Solution {
    public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
        int n = nums.length;
        TreeSet<Long> treeSet = new TreeSet<>();
        for (int i = 0; i < n; i++) {
            //取出当前元素的值为u
            Long u = nums[i] * 1L;
            // 从 treeSet 中找到小于等于 u 的最大值（小于等于 u 的最接近 u 的数）
            Long l = treeSet.floor(u);
            // 从 treeSet 中找到大于等于 u 的最小值（大于等于 u 的最接近 u 的数）
            Long r = treeSet.ceiling(u);

            /**
             * 只需判断l和r是否满足abs(nums[i] - nums[j])
             * 若不满足说明滑动窗口内其它元素也一定不满足
             */
            if (l != null && u - l <= t) {
                return true;
            }
            if (r != null && r - u <= t) {
                return true;
            }

            /**
             * 维护一个大小等于k的滑动窗口，滑动窗口中的数都要添加到treeSet中
             * 这样做的好处就是，treeSet中取出的元素必然满足abs(i - j) <= k的条件，我们只需判断是否满足abs(nums[i] - nums[j])就好了
             */
            treeSet.add(u);
            if (i >= k) {
                treeSet.remove(nums[i - k] * 1L);
            }
        }
        return false;
    }
}
```


### [58. 日程表](https://leetcode-cn.com/problems/fi9suh/)

```java
class MyCalendar {

    TreeMap<Integer, Integer> treeMap;

    public MyCalendar() {
        treeMap = new TreeMap<>();
    }
    
    public boolean book(int start, int end) {
        //在treeMap中找到小于等于start的最近的日程
        //若这个日程的end大于start，说明时间有冲突
        Map.Entry<Integer, Integer> event = treeMap.floorEntry(start);
        if (event != null && event.getValue() > start) {
            return false;
        }

        //在treeMap中找到大于等于start的最近的日程
        //若这个日程的start小于end，说明时间有冲突
        event = treeMap.ceilingEntry(start);
        if (event != null && event.getKey() < end) {
            return false;
        }

        //把当前日程存入treeMap中并返回true
        treeMap.put(start, end);
        return true;
    }
}

/**
 * Your MyCalendar object will be instantiated and called as such:
 * MyCalendar obj = new MyCalendar();
 * boolean param_1 = obj.book(start,end);
 */
```



## 堆

### [59. 数据流的第K大数值](https://leetcode-cn.com/problems/jBjn9C/)

```java
class KthLargest {
    //优先队列，按照从大到小的顺序存储数据流中的数字
    Queue<Integer> queue;
    int k;

    public KthLargest(int k, int[] nums) {
        this.k = k;
        queue = new PriorityQueue<>();
        for (int num : nums) {
            add(num);
        }
    }
    
    public int add(int val) {
        //把val入队，优先队列会自动维护其中元素的顺序
        queue.offer(val);
        //如果队列中元素个数大于k，就一直出队到元素个数等于k为止
        if (queue.size() > k) {
            queue.poll();
        }
        //当队列中元素个数等于k时，最大的k-1个数都出队了，此时队头元素就是第k大个元素
        return queue.peek();
    }
}

/**
 * Your KthLargest object will be instantiated and called as such:
 * KthLargest obj = new KthLargest(k, nums);
 * int param_1 = obj.add(val);
 */
```


### [60. 出现频率最高的k个数字](https://leetcode-cn.com/problems/g5c51o/)

```java
class Solution {
    public int[] topKFrequent(int[] nums, int k) {
        //map:以"数---出现次数"的形式保存每个数在数组中的出现次数
        Map<Integer, Integer> frequencyForNum = new HashMap<>();
        for (int num : nums) {
            frequencyForNum.put(num, frequencyForNum.getOrDefault(num, 0) + 1);
        }
        //初始化一些桶(ArrayList对象数组)
        //每个桶都是ArrayList
        //buckets数组的索引代表了出现次数
        //索引指向的ArrayList里存放的是出现次数为该索引的所有数
        List<Integer>[] buckets = new ArrayList[nums.length + 1];
        //扫描map
        //最后面的桶存放的是出现频率最高的数
        //最前的桶存放的是出现频率最低的数
        for (int key : frequencyForNum.keySet()) {
            //frequency用于存储每个数的出现次数
            int frequency = frequencyForNum.get(key);
            //若出现频率对应的ArrayList为null
            //则需要把它初始化
            if (buckets[frequency] == null) {
                buckets[frequency] = new ArrayList<>();
            }
            //把数加入到存放到对应出现频率的ArrayList中
            buckets[frequency].add(key);
        }
        List<Integer> topK = new ArrayList<>();
        //从后往前遍历桶,因为最后面的桶存放的是出现频率最高的数
        //注意条件topK.size() < k,因为topK存储前k个高频元素,它的大小应该小于k
        for (int i = buckets.length - 1; i >= 0 && topK.size() < k; i--) {
            //若当前桶为null,说明没有数频率为i,直接进行下一轮循环
            if (buckets[i] == null) {
                continue;
            }
            //出现频率为i的数可能有多个
            //当topK中还有剩余位置k - topK.size()可以放下所有出现频率为i的数时
            //就把出现频率为i的数都加入topK
            if (buckets[i].size() <= (k - topK.size())) {
                topK.addAll(buckets[i]);
            } else {
                //当出现频率为i的数太多,只有一部分能放入topK中时
                //便放入k - topK.size()个(正好等于topK的剩余空间)
                topK.addAll(buckets[i].subList(0, k - topK.size()));
            }
        }
        //把topK元素放到数组中并返回
        int[] res = new int[k];
        for (int i = 0; i < k; i++) {
            res[i] = topK.get(i);
        }
        return res;
    }
}
```


### [61. 和最小的k个数对](https://leetcode-cn.com/problems/qn8gGX/)

```java
class Solution {
    public List<List<Integer>> kSmallestPairs(int[] nums1, int[] nums2, int k) {
        //这个优先队列是按照数对和的大小进行排序的，小的数对和排在前面
        Queue<int[]> queue = new PriorityQueue<>((a, b) -> nums1[a[0]] + nums2[a[1]] - nums1[b[0]] - nums2[b[1]]);
        //set用于去除重复的数对和
        Set<String> set = new HashSet<>();
        queue.offer(new int[]{0, 0});
        //ans保存最终返回的结果
        List<List<Integer>> ans = new ArrayList<>();
        //找前k小的数对和，且两个数组都是经过排序的，所以循环最多进行k次
        while (k-- > 0 && queue.size() > 0) {
            //从优先队列中取出最小和的数对放到ans中
            int[] pair = queue.poll();
            ans.add(Arrays.asList(nums1[pair[0]], nums2[pair[1]]));
            /**
             * 因为两个数组都已排序，所以下一个最小数对要么是[nums1中的下标后移1位, nums2中下标不变]，要么是[nums1中下标不变, nums2中的下标后移1位]
             */
            if (pair[0] + 1 < nums1.length) {
                String key = String.valueOf(pair[0] + 1) + "_" + String.valueOf(pair[1]);
                if (set.add(key)) {
                    queue.offer(new int[]{pair[0] + 1, pair[1]});
                }
            }
            if (pair[1] + 1 < nums2.length) {
                String key = String.valueOf(pair[0]) + "_" + String.valueOf(pair[1] + 1);
                if (set.add(key)) {
                    queue.offer(new int[]{pair[0], pair[1] + 1});
                }
            }
        }
        return ans;
    }
}
```


## 前缀树

### [62. 实现前缀树](https://leetcode-cn.com/problems/QC3q1f/)

```java
class Trie {
    //每个节点包含
    //指向子节点的指针数组
    private Trie[] childs;
    //isEnd表示该节点是否为某个字符串的结尾
    private boolean isEnd;

    /** Initialize your data structure here. */
    public Trie() {
        childs = new Trie[26];
        isEnd = false;
    }
    
    /** Inserts a word into the trie. */
    public void insert(String word) {
        Trie node = this;
        //遍历字符串
        for (int i = 0; i < word.length(); i++) {
            char ch = word.charAt(i);
            int index = ch - 'a';//获取当前字符对应的小写字母下标
            //若对应的小写字母下标为null，说明子节点不存在，new一个子节点
            if (node.childs[index] == null) {
                node.childs[index] = new Trie();
            }
            //node指向子节点
            node = node.childs[index];
        }
        //遍历完成后，把最后一个节点isEnd置为true
        node.isEnd = true;
    }
    
    /** Returns if the word is in the trie. */
    public boolean search(String word) {
        //若word存在于trie树中且最后一个字符对应的节点isEnd为true，就返回true
        Trie node = searchPrefix(word);
        return node != null && node.isEnd;
    }
    
    /** Returns if there is any word in the trie that starts with the given prefix. */
    public boolean startsWith(String prefix) {
        return searchPrefix(prefix) != null;
    }

    //寻找前缀prefix
    //若prefix不存在于trie树中则返回null
    //若存在则返回prefix最后一个字符对应的节点
    private Trie searchPrefix(String prefix) {
        Trie node = this;
        for (int i = 0; i < prefix.length(); i++) {
            char ch = prefix.charAt(i);
            int index = ch - 'a';
            //这里代码和插入基本一样，区别是这里判断子节点为null就返回null
            if (node.childs[index] == null) {
                return null;
            }
            node = node.childs[index];
        }
        return node;
    }
}

/**
 * Your Trie object will be instantiated and called as such:
 * Trie obj = new Trie();
 * obj.insert(word);
 * boolean param_2 = obj.search(word);
 * boolean param_3 = obj.startsWith(prefix);
 */
```


### [63. 替换单词](https://leetcode-cn.com/problems/UhWRSj/)

自己写了半天老是错的，还不如直接抄[大佬题解](https://leetcode-cn.com/problems/UhWRSj/solution/zui-yi-yu-li-jie-de-javajie-fa-shi-jian-vcx4e/)

```java
class Solution {

    //前缀树数据结构是一个字符多叉树，用一个数组来保存其子节点， isValid用来标记截止到该节点是否为一个完整的单词
    class TrieNode{
        TrieNode[] kids;
        boolean isValid;
        public TrieNode(){
            kids = new TrieNode[26];
        }
    }

    TrieNode root = new TrieNode();

    public String replaceWords(List<String> dictionary, String sentence) {
        String[] words = new String[dictionary.size()];
        for(int i = 0; i < words.length; ++i) words[i] = dictionary.get(i);
        //建树过程
        for(String word : words){
            insert(root, word);
        }
        String[] strs = sentence.split(" ");
        for(int i = 0; i < strs.length; ++i){
            //如果可以在树中找到对应单词的前缀，那么将这个单词替换为它的前缀
            if(search(root, strs[i])){
                strs[i] = replace(strs[i], root);
            }
        }
        //用StringBuilder来把字符串数组还原成原字符串句子的转换目标字符串
        StringBuilder sb = new StringBuilder();
        for(String s : strs){
            sb.append(s).append(" ");
        }
        sb.deleteCharAt(sb.length()-1);
        return sb.toString();
    }

    //建前缀树模版
    public void insert(TrieNode root, String s){
        TrieNode node = root;
        for(char ch : s.toCharArray()){
            if(node.kids[ch - 'a'] == null) node.kids[ch - 'a'] = new TrieNode();
            node = node.kids[ch - 'a'];
        }
        node.isValid = true;
    }

    //查询是否存在传入的字符串的前缀
    public boolean search(TrieNode root, String s){
        TrieNode node = root;
        for(char ch : s.toCharArray()){
            if(node.isValid == true) break;
            if(node.kids[ch - 'a'] == null) return false;
            node = node.kids[ch - 'a'];
        }
        return true;
    }

    //将传入的字符串替换为它在前缀树中的前缀字符串
    public String replace(String s, TrieNode root){
        TrieNode node = root;
        StringBuilder sb = new StringBuilder();
        for(char ch : s.toCharArray()){
            if(node.isValid || node.kids[ch - 'a'] == null) break;
            node = node.kids[ch - 'a'];
            sb.append(ch);
        }
        return sb.toString();
    }
}
```


## [64. 神奇的字典](https://leetcode-cn.com/problems/US1pGT/)

```java
class MagicDictionary {

    //单词长度 -> 存储同样长度单词的list
    Map<Integer, List<String>> map;

    /** Initialize your data structure here. */
    public MagicDictionary() {
        map = new HashMap<>();
    }
    
    public void buildDict(String[] dictionary) {
        //把dictionary中的单词都存放在map中，若没有对应的数组就先创建
        for (String word : dictionary) {
            int len = word.length();
            if (!map.containsKey(len)) {
                map.put(len, new ArrayList<>());
            }
            map.get(len).add(word);
        }
    }
    
    public boolean search(String searchWord) {
        int len = searchWord.length();
        //如果没有单词与searchWord长度相同，就返回false
        if (!map.containsKey(len)) {
            return false;
        }
        for (String word : map.get(len)) {
            //count记录当前word和searchWord字符不同的个数
            int count = 0;
            for (int i = 0; i < len; i++) {
                if (word.charAt(i) != searchWord.charAt(i)) {
                    count++;
                }
                if (count > 1) {
                    break;
                }
            }
            //若存在单词与searchWord之间只相差一个字符，返回true
            if (count == 1) {
                return true;
            }
        }
        //默认情况下返回false
        return false;
    }
}

/**
 * Your MagicDictionary object will be instantiated and called as such:
 * MagicDictionary obj = new MagicDictionary();
 * obj.buildDict(dictionary);
 * boolean param_2 = obj.search(searchWord);
 */
```


### [65. 最短的单词编码](https://leetcode-cn.com/problems/iSwD2y/)

```java
class Solution {
    public int minimumLengthEncoding(String[] words) {
        int len = 0;
        Trie trie = new Trie();
        /**
         * 把单词按照长度从大到小进行排序
         * 如果先把短的单词（长单词的前缀）插入Trie，长的单词后插入，就会增加助记字符串的长度了
         * 所以必须先插入长的单词，再插入短的前缀，这样能使得助记字符串尽可能短
         */
        Arrays.sort(words, (s1, s2) -> s2.length() - s1.length());
        for (String word : words) {
            len += trie.insert(word);
        }
        return len;
    }

    class Trie {
        TrieNode root;

        public Trie() {
            root = new TrieNode();
        }

        public int insert(String word) {
            TrieNode cur = root;
            boolean isNew = false;
            /**
             * 倒序往Trie中插入单词
             * 若某个单词是已经插入的单词的后缀，倒序插入就为前缀了
             * 这就是用Trie树的原因
             */
            for (int i = word.length() - 1; i >= 0; i--) {
                int c = word.charAt(i) - 'a';
                if (cur.child[c] == null) {
                    isNew = true;
                    cur.child[c] = new TrieNode();
                }
                cur = cur.child[c];
            }
            //如果是新单词，就返回word.length() + 1
            //1指的是"#"符号的长度
            return isNew ? word.length() + 1 : 0;
        }
    }

    class TrieNode {
        char val;
        TrieNode[] child = new TrieNode[26];

        public TrieNode() {}
    }
}
```


### [66. 单词之和](https://leetcode-cn.com/problems/z1R5dt/)

```java
class MapSum {

    class Trie {
        int val;
        Trie[] child = new Trie[26];
    }

    Trie root;

    /** Initialize your data structure here. */
    public MapSum() {
        root = new Trie();
    }
    
    public void insert(String key, int val) {
        Trie node = root;
        for (int i = 0; i < key.length(); i++) {
            int index = key.charAt(i) - 'a';
            if (node.child[index] == null) {
                node.child[index] = new Trie();
            }
            node = node.child[index];
        }
        /**
         * 插入新的key或是覆盖原来的key，都要更新val
         */
        node.val = val;
    }
    
    public int sum(String prefix) {
        int ans = 0;
        Trie node = root;
        for (int i = 0; i < prefix.length(); i++) {
            int index = prefix.charAt(i) - 'a';
            if (node.child[index] == null) {
                node.child[index] = new Trie();
            }
            node = node.child[index];
        }
        /**
         * 搜索到前缀的最后一个字母所在的node
         * 从这个node开始求所有以prefix为前缀的key的val值的和
         */
        return dfs(node);
    }

    //递归，从node开始往下求所有val值的和
    private int dfs(Trie node) {
        if (node == null) {
            return 0;
        }
        int ans = 0;
        for (int i = 0; i < 26; i++) {
            if (node.child[i] != null) {
                //递归地求每个子树的val值的和
                ans += dfs(node.child[i]);
            }
        }
        return ans + node.val;
    }
}

/**
 * Your MapSum object will be instantiated and called as such:
 * MapSum obj = new MapSum();
 * obj.insert(key,val);
 * int param_2 = obj.sum(prefix);
 */
```


### [67. 最大的异或](https://leetcode-cn.com/problems/ms70jA/)

```java
class Solution {
    Trie root = new Trie();
    static final int HIGH_BIT = 30;

    public int findMaximumXOR(int[] nums) {
        int n = nums.length;
        int x = 0;
        for (int i = 1; i < n; i++) {
            add(nums[i - 1]);
            x = Math.max(x, check(nums[i]));
        }
        return x;
    }

    /**
     * 把num以二进制位的形式插入Trie树中
     * @param num
     */
    public void add(int num) {
        Trie cur = root;
        for (int k = HIGH_BIT; k >= 0; k--) {
            int bit = (num >> k) & 1;
            if (bit == 0) {
                if (cur.left == null) {
                    cur.left = new Trie();
                }
                cur = cur.left;
            } else {
                if (cur.right == null) {
                    cur.right = new Trie();
                }
                cur = cur.right;
            }
        }
    }

    /**
     * 计算num和此前插入Trie树的数字的异或运算的最大结果
     * 思想：使尽可能多的位为1
     * @param num
     * @return
     */
    public int check(int num) {
        Trie cur = root;
        int x = 0;
        for (int k = HIGH_BIT; k >= 0; k--) {
            int bit = (num >> k) & 1;
            if (bit == 0) {
                /**
                 * 如果num当前位为0，需要与1进行异或运算以得到1
                 * 所以优先看右子树，若为null则只能看左子树了（当前位为0）
                 */
                if (cur.right != null) {
                    cur = cur.right;
                    x = x * 2 + 1;
                } else {
                    cur = cur.left;
                    x = x * 2;
                }
            } else {
                /**
                 * 如果num当前位为1，需要与0进行异或运算以得到1
                 * 所以优先看左子树，若为null则只能看右子树了（当前位为0）
                 */
                if (cur.left != null) {
                    cur = cur.left;
                    x = x * 2 + 1;
                } else {
                    cur = cur.right;
                    x = x * 2;
                }
            }
        }
        return x;
    }

    /**
     * 这个前缀树比较特殊，只有左右两个节点，分别表示0或1
     * 从根节点到下面的节点表示了nums中某个数字的二进制位的情况（从一个整数最左侧的位开始）
     */
    class Trie {
        /**
         * 左子树指向表示0的子节点
         * 右子树指向表示1的子节点
         */
        Trie left = null;
        Trie right = null;
    }
}
```


## 二分查找

### [68. 查找插入位置](https://leetcode-cn.com/problems/N6YdxV/)

```java
class Solution {
    public int searchInsert(int[] nums, int target) {
        int n = nums.length;
        int left = 0, right = n - 1, ans = n;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (target <= nums[mid]) {
                ans = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return ans;
    }
}
```


### [69. 山峰数组的顶部](https://leetcode-cn.com/problems/B1IidL/)

```java
class Solution {
    public int peakIndexInMountainArray(int[] arr) {
        int n = arr.length;
        int left = 1, right = n - 2, ans = 0;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (arr[mid] > arr[mid + 1]) {
                ans = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return ans;
    }
}
```


### [70. 排序数组中只出现一次的数字](https://leetcode-cn.com/problems/skFtm2/)

```java
class Solution {
    public int singleNonDuplicate(int[] nums) {
        //初始化首尾指针
        int low=0, high=nums.length-1;
        //因为最后是把区间缩小到只包含单一元素(一个元素),并返回这个元素
        //low最终是等于high的
        //所以循环终止条件不是<=
        while (low<high) {
            int mid = low + (high-low)/2;
            //这个操作保证mid为偶数索引
            //同时也保证了mid之前的元素有偶数个
            if (mid%2==1) {
                mid--;
            }
            //若相同元素在mid右侧,说明单一元素一定在mid后面
            if (nums[mid]==nums[mid+1]) {
                low = mid+2;
            }else if (nums[mid]!=nums[mid+1]) {
                //若相同元素不在mid右侧,则单一元素要么是mid要么是mid之前的元素
                high = mid;
            }
        }
        return nums[low];
        //思路:数组被num[mid],num[mid+1]分为两部分:单一元素一定在奇数的那一部分,因为没有元素与它组成一队
    }
}
```



### [71. 按权重生成随机数](https://leetcode-cn.com/problems/cuyjEf/)

```java
class Solution {
    int[] pre;
    int total;

    public Solution(int[] w) {
        pre = new int[w.length];
        pre[0] = w[0];
        for (int i = 1; i < w.length; i++) {
            pre[i] = pre[i - 1] + w[i];
        }
        total = Arrays.stream(w).sum();
    }
    
    public int pickIndex() {
        int x = (int) (Math.random() * total) + 1;
        return binarySearch(x);
    }

    private int binarySearch(int x) {
        int low = 0, high = pre.length - 1;
        while (low < high) {
            int mid = low + (high - low) / 2;
            if (pre[mid] < x) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        return low;
    }
}

/**
 * Your Solution object will be instantiated and called as such:
 * Solution obj = new Solution(w);
 * int param_1 = obj.pickIndex();
 */
```

