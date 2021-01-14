import numpy as np
from PIL import Image
import json


def main():
    # recognition arguments
    # text string, Bernoulli parameter, enable/disable test info:
    # 0, 0-8, 0-1, 0-1
    # 1, 0-5, 0-1, 0-1
    # 2, 0-5, 0-1, 0-1
    d = {0: [0, 1, 2, 3, 4, 5, 6, 7, 8], 1: [0, 1, 2, 3, 4, 5], 2: [0, 1, 2, 3, 4, 5]}
    for string in list(d.keys()):
        for probability in d[string]:
            recognition5_1(string, probability)


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

    pix = np.array(Image.open(folder_name + '/input/' + image_name)).T
    width, height = pix.shape
    width = width // n
    if testid:
        print("width, height of etalons:\n\t", width, ",", height)

    f = open(folder_name + '/frequencies.json')
    freq = json.load(f)

    keys = []
    for i in range(26):
        keys.append(chr(ord('a') + i))
    keys.append('space')

    keys_img = {}
    for k in keys:
        keys_img[k] = np.array(Image.open(folder_name + '/alphabet/' + k + '.png')).T

    keys[len(keys) - 1] = ' '
    keys_img[' '] = keys_img.pop('space')
    freq_keys = list(freq.keys())
    if testid:
        print(keys)
        # print(keys_img)

    prob_total = {}
    for k in keys:
        prob_total[k] = 0
        for kk in keys:
            if k + kk in freq_keys:
                prob_total[k] += freq[k + kk]
            if kk + k in freq_keys:
                prob_total[k] += freq[kk + k]
    prob = {}
    for key in freq:
        prob[key] = freq[key] / prob_total[key[0]]

    if testid:
        print(freq)
        print(prob)
        print(max(prob, key=prob.get), prob[max(prob, key=prob.get)])

    def draw_string(x, n, w, h):
        for i in range(h):
            s = ""
            for j in range(w * n):
                s += str(int(x[j, i]))
            print(s)
        print("\n")

    if testid:
        draw_string(pix, n, width, height)

    def draw_template(temp, h, w):
        for i in range(h):
            s = ""
            for j in range(w):
                s += str(int(temp[j, i]))
            print(s)
        print("\n")

    if testid:
        for i in range(1):
            draw_template(keys_img[keys[i]], height, width)

    ########################################################################

    def q(k1, k2, s, x, p, h=height, w=width):
        qq = 0
        eps = 1e-222
        if k1 + k2 not in freq_keys or prob[k1 + k2] == 0:
            return -np.inf
        xx = x[(s * w):((s + 1) * w), :]
        kk = keys_img[k2][:, :]
        if p * (1 - p) != 0:
            qq = sum(sum(xx ^ kk)) * np.log(p) + sum(sum(1 ^ xx ^ kk)) * np.log(1 - p)
        if p == 0:
            qq = sum(sum(xx ^ kk)) * np.log(eps)
        if p == 1:
            qq = sum(sum(1 ^ xx ^ kk)) * np.log(eps)
        if k1 + k2 in freq_keys:
            return qq + np.log(prob[k1 + k2])

    if testid:
        print(q('e', ' ', 12, pix, p))
        print(q('e', 'd', 12, pix, p))
        print(q('d', 't', 12, pix, p))

    func = []
    for i in range(n):
        fii = {}
        if i == 0:
            for k in keys:
                fii[k] = np.max([q(k, kk, n - 1 - i, pix, p) for kk in keys])
        if 0 < i < n - 1:
            for k in keys:
                fii[k] = np.max([q(k, kk, n - 1 - i, pix, p) + func[i - 1][kk] for kk in keys])
        if i == n - 1:
            for k in [' ']:
                fii[k] = np.max([q(k, kk, n - 1 - i, pix, p) + func[i - 1][kk] for kk in keys])
        func.append(fii)
        if testid:
            print(func[len(func) - 1])

    solution = []
    string = ""
    for i in range(n):
        rii = ""
        if i == 0:
            rii = keys[int(np.argmax([q(' ', kk, i, pix, p) + func[n - 2 - i][kk] for kk in keys]))]
        if 0 < i < n - 1:
            rii = keys[int(np.argmax([q(solution[i - 1], kk, i, pix, p) + func[n - 2 - i][kk] for kk in keys]))]
        if i == n - 1:
            rii = keys[int(np.argmax([q(solution[i - 1], kk, i, pix, p) for kk in keys]))]
        solution.append(rii)
        string += str(rii)

    answer = "Bernoulli parameter: " + str(p) + "\n" \
             + "recognised string: '" + string + "'\n" \
             + "original string:   '" + name + "'\n"
    print(answer)
    return answer


if __name__ == '__main__':
    main()
