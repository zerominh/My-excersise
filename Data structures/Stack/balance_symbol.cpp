#include <iostream>
#include <stack>
#include <cstring>
using namespace std;

bool areParanthesisBalance(string exp)
{
	stack<char> s;
	int len_exp = exp.size();
	for(int i = 0; i < len_exp; ++i)
	{
		if(exp[i] == '(' || exp[i] == '[' || exp[i] == '{')
		{
			s.push(exp[i]);
		}
		else
		{
			if(s.empty())
			{
				return false;
			}
			switch(exp[i])
			{
				case ')':
					if(s.top() != '(')
					{
						return false;
					}
					s.pop();
					break;
				case ']':
					if(s.top() != '[')
					{
						return false;
					}
					s.pop();
					break;	
				case '}':
					if(s.top() != '{')
					{
						return false;
					}
					s.pop();
					break;	

			}
		}
	}
	if(s.empty())
	{
		return true;
	}
	else return false;
	
}
int main()
{
	string s;
	cin >> s;
	if(areParanthesisBalance(s))
	{
		cout << "True";
	}
	else cout << "False";
	return 0;

}