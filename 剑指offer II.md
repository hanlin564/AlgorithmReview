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

