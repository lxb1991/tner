from os import path
import pickle


def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word


def load_data_from_conll(file_path):
    conll_data = []
    conll_sentence = []
    with open(file_path) as fi:
        for lines in fi.readlines():
            line_text = lines.strip()
            if line_text:
                pairs = line_text.split()
                word = normalize_word(pairs[0])
                tag = pairs[1]
                conll_sentence.append((word, tag))
            else:
                conll_data.append(conll_sentence)
                conll_sentence = []
    return conll_data


def load_label_from_conll(file_path):
    label_set = set()
    with open(file_path) as fi:
        for lines in fi.readlines():
            if lines.strip():
                label = lines.strip().split(" ")[1]
                label_set.add(label)
    return label_set


def load_geo_name(geo_name_path):
    geo_name = pickle.load(open(geo_name_path, 'rb'))
    print("location gazetteer contains location num {}".format(len(geo_name)))
    return geo_name


def load_person_name(person_name_path):
    person_name = pickle.load(open(person_name_path, 'rb'))
    print("person name gazetteer contains person name num {}".format(len(person_name)))
    return person_name


def load_multi_data_by_path(config_path):
    model_matrix = {}
    model_tasks = {}
    model_domains = {}
    model_coefficient = {}
    with open(config_path) as fi:
        for lines in fi.readlines():
            if lines.startswith("#"):
                continue
            pairs = lines.strip().split("=")
            model_type = pairs[0]
            model_path = pairs[1]
            model_coef = pairs[2]
            t, d = model_type.split("_")
            if t not in model_tasks:
                model_tasks[t] = len(model_tasks)
            if d not in model_domains:
                model_domains[d] = len(model_domains)
            model_matrix[(t, d)] = model_path
            model_coefficient[(t, d)] = model_coef
    print("load multi model config :")
    print('model matrix : {}'.format(model_matrix))
    print('model task : {}'.format(model_tasks))
    print('model domain : {}'.format(model_domains))
    return model_matrix, model_tasks, model_domains, model_coefficient
