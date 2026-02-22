# Adapted from Johri (2025), by Rani Gera (2025)
# All subject IDs include participants post exclusion

class Subject:
    def __init__(self, subid):
        self.subid = subid  # subject-ID
        self.y = None  # associate a label (or a continuous weight)
        self.x = None  # associate a feature-vector

    @staticmethod
    def get_healthy_controls():
        subids = [2, 3, 4,  7, 10, 12, 13, 19, 20, 22, 23, 25, 30, 31, 32, 34, 36, 37, 38, 41, 46, 48, 49, 50, 51, 52, 53, 54, 57, 58, 59, 60, 61, 63, 64, 65, 66, 69, 70, 72, 74, 76, 77, 80, 81, 82, 83, 84, 85, 87, 88, 89, 91, 92, 93, 94, 95, 96, 97, 98, 99, 101, 109, 112, 114, 115, 116, 117, 118, 119, 120, 125, 126, 128, 129, 130, 131, 134,
                  136, 138, 140, 142, 143, 145, 149, 158, 159, 161, 163, 164, 166, 168, 169, 171, 173, 174, 176, 177, 179, 180, 181, 184, 185, 186, 188, 190, 191, 192, 195, 197, 202, 205, 206, 207, 209, 210, 214, 215, 216, 217, 218, 219, 221, 224, 228, 229, 230, 232, 233, 234, 238, 245, 246, 247, 248, 251, 252, 253, 256, 261, 262, 268, 269, 270]
        return [Subject(subid) for subid in subids]

    @staticmethod
    def get_subids(subjects):
        return sorted([subject.subid for subject in subjects])
