#include <iostream>


using namespace std;

int jumpSearch(int * a, int n, int key)
{

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