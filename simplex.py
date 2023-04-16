import numpy as np
import sys


class Problem:
    def __init__(self, path, fase: int = 2):
        self.fase = fase
        # no necessito les submatrius pq si a=np.array(...) puc accedir amb una llista de index, que sera la b o la n o la q o la B(p), depen del moment
        if fase == 2:

            print("Llegint les dades")

            # creem la instancia fase 1, aquesta al ser creada es pasada a
            # la funcio per obtenir els resultats per la fase 2
            sf1 = Problem(path, fase=1)
            # com abans (fase 1) no necessitavem c no l haviem llegit, el llegim ara
            with open(path, 'r') as f:
                text = f.read()
                rows = text.split('\n')
                k = 0
                if rows[k][0] == 'c':
                    k += 1
                    c = rows[k].split()
                    self.C = np.array(c).astype(float)
                else:
                    raise Exception('ERROR al llegir C en FASE II')

            self.A, self.idxN, self.idxB, self.x, self.invB = sf1.get_sbf()
            print('FASE II')

        elif fase == 1:
            print('FASE I')

            # llegim A i b del fitxer input i utilitzem la info per construir les matrius
            self.readr()

            # passem la instancia a la funcio per resoldre la fase 1
            simplex_solver(self)

            # comprovem si el problema es factible o
            # tenim degeneracio i hem de permutar una variable artificial amb valor 0 per una no basica no artificial
            # si es factible
            self.check_fact()
        else:
            raise Exception('Fase no existent: \n fases possibles: {1, 2}')

        self.r = np.array([0 for _ in range(len(self.idxN))])
        # self.idxB = []
        # self.idxN = []
        # self.x = np.array([])
        # self.A = np.array([])
        # self.B = np.array([])
        # self.invB = np.array([])
        # self.C = np.array([])
        self.db = np.array([0 for _ in range(len(self.idxB))])
        self.p = -1
        self.q = -1
        self.o = float('inf')
        self.z = np.matmul(self.C, self.x)

    def readr(self):

        # metode encarregat de llegir A i b i construir per la fase 1:
        # x, z, db, inversa de B, C
        # conjunt de les basiques i el de les no basiques
        # C i A
        with open(path, 'r') as f:
            text = f.read()
            rows = text.split('\n')
            k = 0
            if rows[k][0] == 'c':
                k += 2
            if len(rows[k]) == 0:
                k += 1

            if rows[k][0] != 'A':
                raise Exception('A no trobada')

            k += 1
            self.A = np.array(rows[k].split())
            k += 1

            while len(rows[k]) != 0:
                self.A = np.vstack((self.A, np.array(rows[k].split())))
                k += 1

            self.A = self.A.astype(float)

            k += 1
            if rows[k][0] != 'b':
                raise Exception('b no trobada')
            k += 1
            # utilitzem b per posar valors a les variables artificials
            b = np.array(rows[k].split())
            b = b.astype(float)
            print(f'b (=Xb): {b}')

        self.aux_art = len(self.A[0])
        self.C = np.array([0 for _ in range(self.aux_art)] + [1 for _ in range(len(b))])
        print(f'C: {self.C}')
        self.x = np.array([0 for _ in range(self.aux_art)] + [i for i in b])
        print(f'x: {self.x}')
        self.idxB = [i for i in range(self.aux_art, self.aux_art+len(b))]
        print(f'conjunt variables Bàsiques (B): {[i+1 for i in self.idxB]}')
        self.idxN = [i for i in range(self.aux_art)]
        print(f'conjunt variables no bàsiques (N): {[i+1 for i in self.idxN]}')

        # afegir a A
        for i in range(len(b)):
            self.A = np.append(self.A, [[0] if j != i else [1] for j in range(len(b))], axis=1)

        print(f'A: \n{self.A}')

        # inversa B
        self.invB = np.identity(len(b))
        self.db = np.array([0 for _ in range(len(self.idxB))])
        self.z = np.matmul(self.C, self.x)
        print(f'z: {self.z}')

    def check_fact(self):
        # mirem si es no factible o si cal permutar les degenerades
        if self.z > 0.000001:
            # raise Exception('PROBLEMA LINEAL NO FACTIBLE')

            print('PROBLEMA LINEAL NO FACTIBLE')
            sys.exit('No tot a la vida té solució :/')
        else:
            for i in self.idxB:
                if i >= self.aux_art:
                    if self.x[i] > 0:
                        raise Exception(f'Si z={self.z}, la variable artificial {i+1} hauria de ser 0, no {self.x[i]}')
                    elif self.x[i] == 0:
                        print('DEGENERACIÓ')
                        print('iniciant permutacions')
                        self.perm_var()

    def perm_var(self):
        # com no he trobat cap no he fet la implementacio
        raise Exception('Falta implementar permutacions')

    def get_sbf(self):
        # metode per passar els resultats de fase 1 a fase 2

        # treure artificials de A i idxN i x
        A = self.A.copy()
        A = np.delete(A, list(range(self.aux_art, len(A[0]))), 1)

        idxN = self.idxN.copy()
        idxN = [i for i in idxN if i < self.aux_art]
        x = self.x.copy()
        x = x[:self.aux_art]
        return A, idxN, self.idxB.copy(), x, self.invB.copy()

    def cal_r(self):
        # calcular els costs reduits
        self.r = self.C[self.idxN] - np.matmul(np.matmul(self.C[self.idxB], self.invB), self.A[:, self.idxN])
        # print(f'r: {self.r}')

    def check_r(self) -> bool:
        # comprovem si tots els costs positius son 0>=
        # (si ho son retornem True que parara l'algorimse pq ja tenim optim)
        # i alhora calculem q tenint en compte la regla de bland
        c = 0
        self.q = self.idxN[c]
        for i in self.r:
            # print(f'q+=: {self.q+1}')
            if i < 0:
                self.q = self.idxN[c]
                return False
            c += 1

        return True

    def cal_db(self):
        # calculem les DBF
        self.db = -1 * np.matmul(self.invB, self.A[:, self.q])

        # print(f'db = {self.db}')

    def check_db(self):
        # mirem si db >= [0]
        # si es cert retornem true pq el problema no es acotat i para l algorisme
        for i in self.db:
            if i < 0:
                return False

        return True

    def cal_o(self):
        # calculem theta i q tenint en compte la regla de bland
        # com db esta ordenat igual que idxB, llavors db[i] es com d[idxB[i]]
        self.o = float("inf")
        p = 0
        for i in range(len(self.idxB)):
            if self.db[i] < 0:

                o = -self.x[self.idxB[i]] / self.db[i]

                if o == self.o:
                    # en cas igualtat hem de mirar si index es mes petit per la regla de Bland
                    if self.idxB[i] < self.idxB[self.p]:
                        self.o = o
                        self.p = p

                elif self.o > o:
                    self.o = o
                    self.p = p
            p += 1
        # print(f'Theta: {self.o}')
        # print(f'p: {self.p + 1}')

    def act_var(self):
        #  actualitzacio de variables
        self.x[self.idxB] = self.x[self.idxB] + self.o * self.db
        self.x[self.q] = self.o

        for i in range(len(self.idxN)):
            if self.idxN[i] == self.q:
                self.z = self.z + self.o * self.r[i]
                break

        # print(f'x={self.x}')
        # print(f'z={self.z}')

    def act_idx(self):
        # actualitzem el conjunt de variables basiques i el de no basiques
        B_p = self.idxB[self.p]
        self.idxB[self.p] = self.q  # on hi havia la que treiem posem la que entra
        self.idxN.remove(self.q)
        self.idxN.append(B_p)
        self.idxN.sort()
        return B_p

        # print(f'conjunt B: {[i + 1 for i in self.idxB]}')
        # print(f'conjunt N: {[i + 1 for i in self.idxN]}')

    def act_inv(self):
        # abans de fer actualitzacio inversa la inversa la feia mitjancant numpy
        # B = self.A[:, self.idxB]
        # print(f'B: \n{B}')
        # invB = np.linalg.inv(B)
        # print(invB)

        self.__act_inversa()

        # print(f'invB: \n{self.invB}')

    def __act_inversa(self):
        # actualitzacio inversa
        # calculem E (fem la identitat i canviem la columna p)
        E = np.identity(len(self.invB))
        for i in range(len(self.invB)):
            if i != self.p:
                E[i, self.p] = -self.db[i]/self.db[self.p]

            else:
                E[i, self.p] = -1/self.db[self.p]
        # calculem la nova inversa i actualitzem
        self.invB = np.matmul(E, self.invB)

    def trace(self, iteration, B_p):
        # imprimim la traca, com python comença a contar de 0 fem un +1 en els indexs que imprimim
        print(f'[simplexP] Fase {self.fase} Iteració {iteration}:  q = {self.q+1}, B(p) = {B_p+1}, theta*= {self.o:.5f}, z = {self.z:.5f}')


def SBF_init(path):
    # crear una instancia de Problem pq solucioni la fase 2 (aixo creara un objecte fase 1 per trobar la SBF inicial (si hi ha una))
    return Problem(path, fase=2)


def simplex_solver(s):
    # pas 1 Inicialitzar (fet previament al crear la instancia de la classe Problem)
    iteration = 1

    while True:
        # print(f'FASE {s.fase} \nIteración: {iteration}')

        # pas 2 escollir var entrada
        # print('PAS 2')
        # calculem el costs reduits
        s.cal_r()

        # mirar si tots son positius (si ho son ja tenim solucio optima --> acabar while True) si no ho son escollir variable no basica dentrada
        if s.check_r():
            if s.fase == 2:
                # mostrar per pantalla output corresponent a la solucio optima trobada
                print("\n\n")
                print("Solució òptima trobada:")
                print("\n\n")

                print('z* =', s.z)
                cnt = 1
                print("Aquest valor s'aconsegueix amb:")
                print(f'conjunt variables bàsiques: {[i + 1 for i in s.idxB]}')

                for xi in s.x:
                    print(f'x{cnt}* = {xi:.5f}')
                    cnt += 1

                print("r:", end="   ")
                for ri in s.r:
                    print(f'{ri:.5f}', end="   ")
                print("\n")

            break
        # print(f'q: {s.q+1}')

        # pas 3 calcular DBF
        # print('PAS 3')
        # calcular les direccions basiques factibles
        s.cal_db()

        # mirar si tots son positius (si ho son problema no acotat --> acabar while True)
        if s.check_db():
            print("PROBLEMA NO ACOTAT")
            break

        # pas 4 escollir var sortida i theta
        # print('PAS 4')
        # calcular theta
        s.cal_o()

        # pas 5 actualitzacio + canvi de base
        # print('PAS 5')
        # var
        s.act_var()

        # base (canviar els valors de xb xq An B inversa B)
        B_p = s.act_idx()

        # actualitzacio inversa
        s.act_inv()

        # un cop fet tot mostrem els resultats de la iteració
        s.trace(iteration, B_p)

        # actualitzem la iteració on ens trobem
        iteration += 1
        # input('FINAL ITERACIÓ')


if __name__ == '__main__':
    print('Simplex START')
    path = sys.argv[1]
    # creem una instancia de la fase 2, al ferho es creara una fase 1 i es solucionara per crear la fase 2
    sf2 = SBF_init(path)

    # apliquem la funcio que s encarrega d'executar l algorimse a aquesta instancia
    simplex_solver(sf2)

    print('END')
