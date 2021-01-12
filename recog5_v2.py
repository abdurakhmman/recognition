import numpy as np
from PIL import Image
import json


def main():
    # recognition arguments
    # text string, Bernoulli parameter, enable/disable test info:
    # 0, 0-8, 0-1
    # 1, 0-5, 0-1
    # 2, 0-5, 0-1
    recognition5_1(0, 0)
    # recognition5_1(0, 1)
    # recognition5_1(0, 2)
    # recognition5_1(0, 3)
    # recognition5_1(0, 4)
    # recognition5_1(0, 5)
    # recognition5_1(0, 6)
    # recognition5_1(0, 7)
    # recognition5_1(0, 8)
    #
    # recognition5_1(1, 0)
    # recognition5_1(1, 1)
    # recognition5_1(1, 2)
    # recognition5_1(1, 3)
    # recognition5_1(1, 4)
    # recognition5_1(1, 5)
    #
    # recognition5_1(2, 0)
    # recognition5_1(2, 1)
    # recognition5_1(2, 2)
    # recognition5_1(2, 3)
    # recognition5_1(2, 4)
    recognition5_1(2, 5)


def recognition5_1(nameid=0, probid=7, testid=0):
    names = {"very simple text": [0, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 1],  # 0
             #                    0,   1,   2,    3,   4,    5,   6,   7, 8
             "but thence i learn and find the lesson true drugs poison him that so feil sick of you":  # 1
                        [0.3, 0.4, 0.45, 0.55, 0.6, 0.7],
             #             0,   1,    2,    3,   4,   5
             "i am alone the villain of the earth and feel i am so most":  # 2
                 [0.3, 0.4, 0.45, 0.55, 0.6, 0.65]
             #      0,   1,    2,    3,   4,    5
             }
    names_list = list(names.keys())
    name = names_list[nameid]
    p = names[name][probid]
    n = len(name)
    if testid:
        print('\nnumber of letters:\n\t', n, '\nBernoulli parameter:\n\t', p, '\nstring itself:\n\t', "'" + name + "'")
    folder_name = "recog5_data"
    image_name = name + "_" + str(p) + ".png"

    im = Image.open(folder_name + '/input/' + image_name)
    pix = im.load()
    width, height = im.size
    width = width // n
    if testid:
        print("width, height of etalons:\n\t", width, ",", height)

    for s in range(n):
        for i in range(height):
            for j in range(width):
                pix[j + s * width, i] = pix[j + s * width, i] // 255

    def draw_string(x, n, w, h):
        for i in range(h):
            s = ""
            for j in range(w * n):
                s += str(x[j, i])
            print(s)
        print("\n")

    if testid:
        draw_string(pix, n, width, height)

    f = open(folder_name + '/frequencies.json')
    freq = json.load(f)
    mass_total = 0
    for key in freq:
        mass_total += freq[key]
    for key in freq:
        freq[key] = freq[key] / mass_total
    # print(mass_total)

    keys = []
    for i in range(26):
        keys.append(chr(ord('a') + i))
    keys.append('space')

    keys_img = {}
    for k in keys:
        template = Image.open(folder_name + '/alphabet/' + k + '.png')
        temp = template.load()
        for i in range(height):
            for j in range(width):
                temp[i, j] = temp[i, j] // 255
        keys_img[k] = temp

    keys[len(keys) - 1] = ' '
    keys_img[' '] = keys_img.pop('space')
    freq_keys = list(freq.keys())
    if testid:
        print(keys)
        print(keys_img)

    def draw_template(k, h, w):
        for i in range(h):
            s = ""
            for j in range(w):
                s += str(keys_img[k][j, i])
            print(s)
        print("\n")

    if testid:
        for i in range(3):
            draw_template(keys[i], height, width)

    ########################################################################

    def q(k1, k2, s, x, p, h=height, w=width):
        eps = 1e-220
        qu = 0
        if p * (1 - p) != 0:
            for i in range(h):
                for j in range(w):
                    bit = x[j + s * w, i] ^ keys_img[k2][j, i]
                    qu += bit * np.log(p) + (1 ^ bit) * np.log(1 - p)
        if p == 0:
            for i in range(h):
                for j in range(w):
                    bit = x[j + s * w, i] ^ keys_img[k2][j, i]
                    qu += bit * np.log(eps)
        if p == 1:
            for i in range(height):
                for j in range(width):
                    bit = x[j + s * width, i] ^ keys_img[k2][j, i]
                    qu += (1 ^ bit) * np.log(eps)
        if k1 + k2 not in freq_keys:
            return qu + np.log(eps)
        if k1 + k2 in freq_keys:
            return qu + np.log(freq[k1 + k2])

    if testid:
        print(q('a', ' ', 5, pix, p))
        print(q('z', 't', 5, pix, p))

    func = []
    fnn_1 = {}
    for k in keys:
        fnn_1[k] = np.max([q(k, kk, n - 1, pix, p) for kk in keys])
    func.append(fnn_1)
    if testid:
        print(func[0])

    for i in range(1, n):
        fii = {}
        for k in keys:
            fii[k] = np.max([q(k, kk, n - 1 - i, pix, p) + func[i - 1][kk] for kk in keys])
        func.append(fii)
    if testid:
        print(func[len(func) - 1])

    solution = [keys[int(np.argmax([q(' ', kk, 0, pix, p) + func[n - 2][kk] for kk in keys]))]]
    for i in range(1, n):
        solution.append(keys[int(np.argmax([q(solution[i - 1], kk, i, pix, p) + func[n - i - 1][kk] for kk in keys]))])

    string = ""
    for i in range(len(solution)):
        string += str(solution[i])

    answer = "Bernoulli parameter: " + str(p) + "\n" \
             + "recognised string: '" + string + "'\n" \
             + "original string:   '" + name + "'\n"
    print(answer)
    return answer


if __name__ == '__main__':
    main()
