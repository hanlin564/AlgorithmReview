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

