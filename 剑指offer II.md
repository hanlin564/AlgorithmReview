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