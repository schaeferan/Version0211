import xml.etree.ElementTree as ET
import numpy as np
from scipy.linalg import rq

def process_projection_matrices(projection_matrices):
    num_matrices = projection_matrices.shape[0]

    # Arrays zur Speicherung der Ergebnisse
    Ks = np.zeros((num_matrices, 3, 3), dtype=np.float32)
    Rs = np.zeros((num_matrices, 3, 3), dtype=np.float32)
    ts = np.zeros((num_matrices, 3), dtype=np.float32)
    camtoworlds = np.zeros((num_matrices, 3, 4), dtype=np.float32)

    for i in range(num_matrices):
        # Extrahiere M und p_4 aus der Projektionsmatrix
        P = projection_matrices[i]
        M = P[:, :3]
        p_4 = P[:, 3]

        # Führe die RQ-Zerlegung an M durch
        K, R = rq(M)

        # Sicherstellen, dass die Hauptdiagonalelemente von K positiv sind
        T = np.diag(np.sign(np.diag(K)))
        K = np.dot(K, T)
        R = np.dot(T, R)

        # Berechne den Translationsvektor t
        t = np.linalg.inv(K) @ p_4

        # Speichere K, R und t
        Ks[i] = K
        Rs[i] = R
        ts[i] = t

        # Berechne die [R|T] Matrix und die camtoworld Matrix
        K_inverse = np.linalg.inv(K)
        RT = np.matmul(K_inverse, P)
        R_c2w = np.transpose(RT[:, :3])
        t_expanded = np.expand_dims(RT[:, 3], axis=1)
        result = -np.matmul(R_c2w, t_expanded)
        t_c2w = np.squeeze(result, axis=1)

        # Kombiniere R_c2w und t_c2w zu camtoworld
        camtoworlds[i] = np.hstack((R_c2w, t_c2w[:, np.newaxis]))

    return Ks, camtoworlds
def extract_projection_matrices_DRR(xml_file_path):
    """
    Extrahiert die Projektionsmatrizen aus einer XML-Datei und gibt sie als 3D-Array zurück.

    Args:
    - xml_file_path (str): Pfad zur XML-Datei, die die Projektionsmatrizen enthält.

    Returns:
    - projection_matrices_array (np.ndarray): 3D-Array der Projektionsmatrizen (Anzahl der Matrizen, 3, 4).
    """

    # XML-Datei parsen
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # Liste für die Projektionsmatrizen
    projection_matrices = []

    # Projektionsmatrizen aus dem XML extrahieren
    for i in range(400):
        matrix_element = root.find(f'./ElementList/PROJECTION_MATRICES/M{i}')
        if matrix_element is not None:
            matrix_values = list(map(float, matrix_element.text.split()))
            projection_matrices.append(matrix_values)

    # In ein 3D-Array umwandeln (Anzahl der Matrizen, Zeilen, Spalten)
    projection_matrices_array = np.array(projection_matrices).reshape(400, 3, 4)

    return projection_matrices_array

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

            # Konvertieren Sie die Liste von Matrizen in ein 3D-Array
            matrices_3d = np.array(matrices)

            # Zum Speichern des 3D-Arrays in einer Datei (z.B. im .npy-Format)
            # np.save('/Pfad/zum/Speichern/der/Arrays.npy', matrices_3d)

        else:
            print("Das Element 'projectionMatrices' wurde nicht gefunden.")

        return matrices_3d  # Geben Sie das 3D-Array zurück

    except Exception as e:
        print("Fehler beim Parsen der XML-Datei:", e)

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







