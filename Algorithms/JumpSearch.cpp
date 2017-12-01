#include <iostream>
#include <cmath>

using namespace std;

int jumpSearch(int * a, int n, int key)
{
	int step = (int)sqrt(n);
	for (int i = 0; i < n; i += step)
	{
		if(a[i] >= key)
		{
			int prev = i - step;
			for(int j = i; j >= prev; --j)
			{
				if(a[j] == key)
				{
					return j;
				}
			}
			return -1;
		}
	}
	return -1;
}


int main()
{
	int n, key;
	cin >> n >> key;
	int * a = new int[n];
	for(int i = 0; i < n; ++i)
	{
		cin >> a[i];
	}

	cout << jumpSearch(a, n, key);
	return 0;
}