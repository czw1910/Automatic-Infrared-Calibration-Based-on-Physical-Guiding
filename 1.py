

#include <iostream>
#include <string>
using namespace std;

class Solution {
public:
    void reverse(string& s, int start, int end) {
        for (int i = start, j = end; i < j; i++, j--) {
            swap(s[i], s[j]);
        }
    }

    string replaceSpace(string s) {
        int j = 0;
        int sOldSize = s.size();
        for (int i = 0; i < s.size(); i++) {
            if (s[i] == ' ') {
                reverse(s,j,i);
                j = i+1;
            }
b
        }


        return s;
    }
};

int main() {
    // ACM模式输入处理
    string input;

    // 读取输入字符串（包含空格）
    getline(cin, input);  // 使用getline读取整行

    // 处理字符串
    Solution solution;
    string result = solution.replaceSpace(input);

    // 输出结果
    cout << result << endl;

    return 0;
}