from pathlib import Path
from wasabi import msg
import tarfile
import numpy
from tqdm import tqdm
from spacy.vectors import Vectors

def add_vectors(nlp, vectors_loc, name=None):
    vectors_loc = Path(vectors_loc)
    if vectors_loc and vectors_loc.parts[-1].endswith(".npz"):
        nlp.vocab.vectors = Vectors(data=numpy.load(vectors_loc.open("rb")))
        for lex in nlp.vocab:
            if lex.rank:
                nlp.vocab.vectors.add(lex.orth, row=lex.rank)
    else:
        if vectors_loc:
            vectors_data, vector_keys = read_vectors(vectors_loc)
        else:
            vectors_data, vector_keys = (None, None)
        if vector_keys is not None:
            for word in vector_keys:
                if word not in nlp.vocab:
                    lexeme = nlp.vocab[word]
                    lexeme.is_oov = False
        if vectors_data is not None:
            nlp.vocab.vectors = Vectors(data=vectors_data, keys=vector_keys)
    if name is None:
        nlp.vocab.vectors.name = "%s_model.vectors" % nlp.meta["lang"]
    else:
        nlp.vocab.vectors.name = name
    nlp.meta["vectors"]["name"] = nlp.vocab.vectors.name
    return nlp


def read_vectors(vectors_loc):
    f = open_file(vectors_loc)
    shape = tuple(int(size) for size in next(f).split())
    vectors_data = numpy.zeros(shape=shape, dtype="f")
    vectors_keys = []
    for i, line in enumerate(tqdm(f)):
        line = line.rstrip()
        pieces = line.rsplit(" ", vectors_data.shape[1])
        word = pieces.pop(0)
        if len(pieces) != vectors_data.shape[1]:
            msg.fail(Errors.E094.format(line_num=i, loc=vectors_loc), exits=1)
        vectors_data[i] = numpy.asarray(pieces, dtype="f")
        vectors_keys.append(word)
    return vectors_data, vectors_keys

def open_file(loc):
    """Handle .gz, .tar.gz or unzipped files"""
    loc = Path(loc)
    if tarfile.is_tarfile(str(loc)):
        return tarfile.open(str(loc), "r:gz")
    elif loc.parts[-1].endswith("gz"):
        return (line.decode("utf8") for line in gzip.open(str(loc), "r"))
    elif loc.parts[-1].endswith("zip"):
        zip_file = zipfile.ZipFile(str(loc))
        names = zip_file.namelist()
        file_ = zip_file.open(names[0])
        return (line.decode("utf8") for line in file_)
    else:
        return loc.open("r", encoding="utf8")