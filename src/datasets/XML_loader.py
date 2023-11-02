import xml.etree.ElementTree as ET
import numpy as np


def parse_projection_matrices(xml_file_path):
    try:
        # XML-Datei parsen
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        # Suchen Sie das Element, das die Matrizen enthält (in diesem Fall 'projectionMatrices')
        projection_matrices_elem = root.find(".//array[@class='edu.stanford.rsl.conrad.geometry.Projection']")

        matrices = []  # Initialisieren Sie die Liste für Matrizen

        if projection_matrices_elem is not None:
            # Extrahieren Sie die Matrizen aus dem XML-Element
            for matrix_elem in projection_matrices_elem.findall('.//string'):
                matrix_string = matrix_elem.text.strip()
                # Entfernen Sie eckige Klammern und Semikolon
                matrix_string = matrix_string.replace('[', '').replace(']', '').replace(';', '')
                # Matrix-String in eine Liste von Listen von Floats umwandeln
                matrix = [[float(x) for x in row.split()] for row in matrix_string.split()]

                # Überprüfen, ob die Matrix die erwartete Form (12, 1) hat
                if len(matrix) == 12 and len(matrix[0]) == 1:
                    # In ein (3, 4)-Array umwandeln
                    reshaped_matrix = np.array(matrix).reshape(3, 4)
                    matrices.append(reshaped_matrix)
                else:
                    print(f"Matrix {len(matrices) + 1} hat nicht die erwartete Form (12, 1).")

            # Zum Speichern der NumPy-Arrays in einer Datei (z.B. im .npy-Format)
            # np.savez('/Pfad/zum/Speichern/der/Arrays', *matrices)

        else:
            print("Das Element 'projectionMatrices' wurde nicht gefunden.")

        return matrices  # Geben Sie die Liste der Matrizen zurück

    except Exception as e:
        print(f"Fehler beim Parsen der XML-Datei: {str(e)}")
        return []


def analyze_xml_file(xml_file_path):
    try:
        # XML-Datei analysieren
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        # Initialisiere ein Dictionary, um die Daten zu speichern
        data_dict = {}

        # Extrahiere und speichere alle Daten außer den Projections-Matrizen
        for void in root.findall(".//void"):
            property_name = void.get("property")
            if property_name != "projectionMatrices":
                if void.find("string") is not None:
                    data = void.find("string").text.strip()
                elif void.find("int") is not None:
                    data = int(void.find("int").text)
                elif void.find("double") is not None:
                    data = float(void.find("double").text)
                elif void.find("boolean") is not None:
                    data = bool(void.find("boolean").text)
                else:
                    data = None

                if data is not None:
                    data_dict[property_name] = data

        # Drucke die Anzahl der Projections-Matrizen
        num_projection_matrices = len(
            root.findall(".//array[@class='edu.stanford.rsl.conrad.geometry.Projection']/void/object/void/string"))
        data_dict["Number of Projection Matrices"] = num_projection_matrices

        return data_dict

    except Exception as e:
        print(f"Fehler beim Analysieren der XML-Datei: {str(e)}")
        return None







