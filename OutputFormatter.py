import numpy as np


class Formatter:

    def format_output(variables, values, cluster, T):
        ##
        L = len(cluster) + 2
        result = np.zeros(T*(L))
        for variable_idx in range(len(variables)):
            val = values[variable_idx]
            name_pieces = Formatter.get_name_pieces(variables[variable_idx])

            if name_pieces[0] == "PG":
                t = int(name_pieces[1])
                idx = t * L + len(cluster)
            elif name_pieces[0] == "PS":
                t = int(name_pieces[1])
                idx = t * L + len(cluster) + 1
            else:
                t = int(name_pieces[2])
                i = int(name_pieces[1])
                idx = (t * L) + i

            result[idx] = val

        return result

    def get_name_pieces(name):
        result = []
        name_pieces = str(name).split("_")
        if len(name_pieces) > 1:
            result.append(name_pieces[0])
            for piece in name_pieces[1:]:
                if piece.isdigit():
                    result.append(piece)
                else:
                    result[0] += "_" + piece
        else:
            result = name

        # test method is not working correctly so this is temp return to old method
        result = name_pieces
        # test

        return result
