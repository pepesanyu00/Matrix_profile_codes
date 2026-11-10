// Reproduce la seleccion de (meta-)diagonales del binario scamp/scrimp:
//   diags = {EZ+1, EZ+1+step, ...} ; shuffle(mt19937(0)) opcional ; Ndiags = size*pct/100
// Imprime, para cada bundle seleccionado (en orden de reparto dinamico), su coste en celdas.
// uso: sample_sel P w step pct shuffle(0/1)
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
using namespace std;
typedef uint64_t ITYPE;
int main(int argc, char **argv) {
  ITYPE P = atoll(argv[1]), w = atoll(argv[2]), step = atoll(argv[3]), pct = atoll(argv[4]);
  int sh = atoi(argv[5]);
  ITYPE ez = (ITYPE)(w * 0.25);
  vector<ITYPE> diags;
  for (ITYPE i = ez + 1; i < P; i += step) diags.push_back(i);
  if (sh) { mt19937 g(0); shuffle(diags.begin(), diags.end(), g); }
  ITYPE n = (ITYPE)diags.size() * pct / 100;
  printf("#total_bundles %zu\n", diags.size());
  for (ITYPE r = 0; r < n; r++) {
    ITYPE d = diags[r], cells = 0;
    for (ITYPE k = 0; k < step && d + k < P; k++) cells += P - (d + k);
    printf("%llu %llu\n", (unsigned long long)d, (unsigned long long)cells);
  }
  return 0;
}
