from matplotlib import pyplot as plt


def char_frequencies(path):
    """
    counts the number of each character in the given file path and returns a normalized key-value dict
    :param path:
    :return:
    """
    chars = {}
    line_count = 0
    with open(path, 'r') as file:
        f = file.readlines()
        for line in f:
            line_count += 1
            for c in line:
                if c in chars.keys():
                    chars[c] += 1
                else:
                    chars[c] = 1

    for k in chars.keys():
        chars[k] /= line_count
    return chars


def char_center_of_mass(path):
    """
    measures the average location of each character in the given file path and returns a normalized key-value dict
    :param path:
    :return:
    """
    chars = {}
    line_count = 0
    with open(path, 'r') as file:
        f = file.readlines()
        for line in f:
            line_count += 1
            for i in range(len(line)):
                c = line[i]
                if c in chars.keys():
                    chars[c] += i
                else:
                    chars[c] = i

    for k in chars.keys():
        chars[k] /= line_count
    return chars


if __name__ == "__main__":
    neg = char_frequencies("neg_A0201.txt")
    pos = char_frequencies("pos_A0201.txt")

    # neg = char_center_of_mass("neg_A0201.txt")
    # pos = char_center_of_mass("pos_A0201.txt")
    #
    keys = sorted(neg.keys())
    keys.remove("\n")
    new_neg = {k: neg[k] for k in keys}
    new_pos = {k: pos[k] for k in keys}
    diff = {k: pos[k] - neg[k] for k in keys}

    plt.bar(new_neg.keys(), new_neg.values())
    plt.title("Negative samples average Amino Acid Position")
    plt.show()
    plt.bar(new_pos.keys(), new_pos.values())
    plt.title("Positive samples average Amino Acid Position")
    plt.show()
    plt.bar(diff.keys(), diff.values())
    plt.title("Difference of samples average Amino Acid Histogram")
    plt.show()
