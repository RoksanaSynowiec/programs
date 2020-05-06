//szyfr

#include <iostream>
#include <fstream>
#include <map>
using namespace std;
int main ()
{	string tab[1000];
	string slowo;
	ifstream plik;
	ofstream odpowiedz;
	plik.open("C:/Users/Roksana/OneDrive/Pulpit/sygnaly.txt");

		for(int i=0; i<1000; i++){
	
			plik>>tab[i];
	}
	
map<char, int> licznik;
int lili[1000];
int max=lili[0];
int index=0;
	for(int j=0; j<1000; j++){

		for(int i=0; i<tab[j].size(); i++){

			licznik[tab[j][i]]++;
			lili[j]=licznik.size();
		}licznik.clear();
	}
	for(int i=1; i<1000; i++){
		if(lili[i]>max){
			max=lili[i];
			index=i;
		}
	}
cout<<tab[index];
	}

	
