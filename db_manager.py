import pickle
from dataclasses import dataclass

@dataclass
class Competiciones:
    nombre: str
    temporada: str
    infantil: bool
    archivo: str

@dataclass
class DataBase:
    competiciones: list[Competiciones]

    def save(self, filename: str):
        with open(filename, 'wb') as f:
                pickle.dump(self, f)
    
    @classmethod
    def load(cls, filename: str):
        with open(filename, 'rb') as f:
            return pickle.load(f)
        
    def add_competicion(self, competicion: Competiciones):
        self.competiciones.append(competicion)

    def get_competicion(self, nombre: str) -> Competiciones:
        for competicion in self.competiciones:
            if competicion.nombre == nombre:
                return competicion
        return None
    

if __name__ == "__main__":
    db = DataBase([])

    # ASOBAL
    db.add_competicion(Competiciones('ASOBAL 21-22', '2021-2022', False, 'partidos_ASOBAL2122.xlsx'))
    db.add_competicion(Competiciones('ASOBAL 22-23', '2022-2023', False, 'partidos_ASOBAL2223.xlsx'))
    db.add_competicion(Competiciones('ASOBAL 23-24', '2023-2024', False, 'partidos_ASOBAL2324.xlsx'))

    # GUERRERAS
    db.add_competicion(Competiciones('GUERRERAS 21-22', '2021-2022', False, 'partidos_Guerreras2122.xlsx'))
    db.add_competicion(Competiciones('GUERRERAS 22-23', '2022-2023', False, 'partidos_Guerreras2223.xlsx'))
    db.add_competicion(Competiciones('GUERRERAS 23-24', '2023-2024', False, 'partidos_Guerreras2324.xlsx'))

    # DHOF
    db.add_competicion(Competiciones('DHOF 22-23', '2022-2023', False, 'partidos_DHOF2223.xlsx'))
    db.add_competicion(Competiciones('DHOF 23-24', '2023-2024', False, 'partidos_DHOF2324.xlsx'))


    db.save('db.pkl')