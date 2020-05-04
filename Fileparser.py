import os

from io import open
import sys

import xml.dom.minidom

import re
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
import RetinaOperations



class document:
    # A class for storing a document and its corresponding list of plagiarism class objects
  def __init__(self, name, plagiarismList):
    self.name = name
    self.plagiarismList = plagiarismList

  def toString(self):
    print(self.name)
    for p in self.plagiarismList:
        string = p.toString()
        print("plagiarism: " + string)

class plagiarism:
    # A class of the plagiarism object, defined by lenght and offset
    def __init__(self, offset, length):
        self.offset = offset
        self.length = length

    def toString(self):
        string = "offset: " + str(self.offset) + " length: " + str(self.length)
        return string






def extract_text_from_document(path, document):
    # extracts raw text from documents
    filename = str(path) + "/" + str(document.name)
    with open(filename, encoding="utf-8", errors='ignore') as f:
        text = f.read()
    return str(text)

def extract_plagiarisms_from_text(text, document):
    # extracts list of plagiarisms from a document
    plagiarisms = [];
    for p in document.plagiarismList:
            start = p.offset
            end  = start + p.length
            plagiarisms.append(text[start:end])
    return plagiarisms

    # Methods for extracting plagiarisms from XML files - from PAN 2011
def extract_plagiarisms_from_files(path):
    """Returns a set of plagiarism annotations from XML files below path."""
    if not os.path.exists(path):
        print("Path not accessible:", path)
        sys.exit(2)
    documentList = [];
    num_files = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
    for i in range(1,int(num_files/2)+1):
        idxLen = len(str(i));
        filename = "suspicious-document0"
        for j in range(4-idxLen):
            filename += "0"
        filename+=str(i)
        docName=filename+".txt"
        xmlName=path+"/"+filename+".xml"
        plagiarismList = extract_plagiarisms_from_file(xmlName)
        documentList.append(document(docName,plagiarismList))
    return documentList


def extract_plagiarisms_from_file(xmlfile):
    """Returns a set of plagiarism annotations from an XML file."""
    doc = xml.dom.minidom.parse(xmlfile)
    plagiarismList = []
    for node in doc.documentElement.childNodes:
        if node.nodeType == xml.dom.Node.ELEMENT_NODE and \
           node.hasAttribute('name') and \
           node.getAttribute('name').endswith("plagiarism"):
            plagiarism = extract_plagiarism_from_node(node)
            if plagiarism:
                plagiarismList.append(plagiarism)
    return plagiarismList


def extract_plagiarism_from_node(xmlnode, ):
    """Returns a plagiarism annotation from an XML feature tag node."""
    if not (xmlnode.hasAttribute('this_offset') and \
            xmlnode.hasAttribute('this_length')):
        return False
    offset = int(xmlnode.getAttribute('this_offset'))
    length = int(xmlnode.getAttribute('this_length'))

    return plagiarism(offset, length)





def findOverlaps(plagiarismList, detectionList):
    # method that finds the number of overlapping characters within our dataset,
    # and looks for fully overlapping and partially overlapping sections
    overlappingChars=0
    fullOverlaps=0
    partialOverlaps=0
    for detection in detectionList:
        dec_start=detection.offset
        dec_end=dec_start+detection.length
        partial=0
        for plagiarism in plagiarismList:
            if (dec_start<=plagiarism.offset and dec_end>=plagiarism.offset or
                dec_start>plagiarism.offset and dec_start<=plagiarism.offset+plagiarism.length):
                    start=max(dec_start,plagiarism.offset)
                    end=min(dec_end,plagiarism.offset+plagiarism.length)
                    overlappingChars+=end-start
                    if(dec_start==plagiarism.offset and dec_end==plagiarism.offset+plagiarism.length):
                        fullOverlaps+=1
                    else:
                        partial = 1
        partialOverlaps+=partial
    return overlappingChars,fullOverlaps, partialOverlaps

def listLength(list):
    # calculates the sum of the lengths of the individual elements in the list
    counter=0
    for l in list:
        counter+=l.length
    return counter

def confusionMatrix(text,plagiarismList, detectionList):
    # calculates precision, recall, accuracy and the overlaps of the detections and actual plagiarisms
    fullLength=len(text)
    plagiarismLength=listLength(plagiarismList)
    detectionLength=listLength(detectionList)
    TPLength,fullOverlaps,partialOverlaps=findOverlaps(plagiarismList,detectionList)
    FPLength=detectionLength-TPLength
    FNLength=plagiarismLength-TPLength
    TNLength=fullLength-TPLength-FPLength-FNLength
    if (detectionLength==0):
        precision=-1
    else:
        precision=TPLength/detectionLength
    if (plagiarismLength == 0):
        recall=-1
    else:
        recall = TPLength / plagiarismLength

    accuracy=(TPLength+TNLength)/fullLength
    return precision,recall,accuracy,fullOverlaps, partialOverlaps


def split_into_sentences(text):
    # splits the text into sentences and also preserves the corresponding starting and ending indices
    startIndices=[]
    endIndices=[]
    corpus=[]
    punkt_param = PunktParameters()
    punkt_param.abbrev_types = set(['dr', 'doc', 'mr', 'mrs', 'prof', 'inc', 'mgr', 'ing', 'st'])
    sentence_splitter = PunktSentenceTokenizer(punkt_param)

    for start, end in sentence_splitter.span_tokenize(text):
        startIndices.append(start)
        endIndices.append(end)
        token = text[start:end]
        corpus.append(token)
    return startIndices, endIndices, corpus

# methods for cleaning the text of non-alphabet characters
def clean_corpus(corpus):
    for i in range(len(corpus)):
        corpus[i]=clean_text(corpus[i]);
    return corpus

def  clean_text(text):

        text = text.lower()
        #text = re.sub(r'\W', ' ', text)
        #text = re.sub(r'\s+', ' ', text)
        #text = re.sub("\d+", "", text)
        text = re.sub("[^a-zA-Z'\d\s:]"," ",text)
        text = re.sub(r'\s+', ' ', text)
        return text

def tag_sentence_indices(corpus):
    # tags sentence indices to know what character offset they occur at - deprecated
    idxTags = [];
    offset=0;
    for i in range(len(corpus)):
        idxTags.append(offset);
        offset+=len(corpus[i]);
    return idxTags;

def merge_corpus(corpus, endIndices):
    # merges the corpus so that sentences containing a single word will get merged with the following ones.
    # we will save the ending indices so we know where each new sentence ends and which merged sentence each sentence belongs to
    newCorpus=[]
    newIndices=[]

    buffer="";
    for i in range(len(corpus)-1):
        if len(corpus[i].split()) > 1:
            newCorpus.append(str(buffer + " " + corpus[i]))
            newIndices.append(endIndices[i])
            buffer=""
        else:
            buffer+=str(corpus[i])
    finalIdx=len(corpus)-1
    finalText=str(buffer+corpus[finalIdx])
    if (finalText.isspace()==False):
        newCorpus.append(str(finalText))
        newIndices.append(endIndices[finalIdx])



    return newCorpus,newIndices



def extract_sentences_into_files(documentList, sourcePath, corpusPath, indexPath):
    # writes the split and merged sentences as well as their corresponding ending indices into a file
    for i in range(len(documentList)):
        text = extract_text_from_document(sourcePath, documentList[i])
        start, end, corpus= split_into_sentences(text)
        corpus=clean_corpus(corpus)

        corpus, endIndices =merge_corpus(corpus,end)

        cPath=corpusPath+"/Corpus"+str(i)+".txt"
        iPAth=indexPath+"/Indices"+str(i)+".txt"
        with open(cPath, 'w', encoding="utf-8", errors='ignore') as f:
            f.truncate(0)
            for j in range(len(corpus)):
                f.write(str(corpus[j]))
                f.write("|")
            f.close()
        with open(iPAth, 'w', encoding="utf-8", errors='ignore') as f:
            f.truncate(0)
            for j in range(len(endIndices)):
                f.write(str(endIndices[j]))
                f.write("|")
            f.close()

    return corpus, endIndices

def get_corpus_from_file(path):
    # returns a list of split sentences corresponding to a document
    with open(path, encoding="utf-8", errors='ignore') as f:
        text=f.read()
        corpus = text.split("|")
        f.close()
    return corpus

def get_fingerprint_from_file(path):
    # returns the list of semantic fingerprints from a file
    printList=[]
    with open(path, encoding="utf-8") as f:
        text = f.read()
        fingerprints = text.split("|")
        for i in range(len(fingerprints)-1):
            fPrint = fingerprints[i].split(",")
            fPrint.pop()
            printList.append(fPrint)
        f.close()
    return printList

def get_clean_fingerprints_from_file(path):
    # returns clean fingerprints from a file for a certain document
    printList = []
    with open(path, encoding="utf-8") as f:
        text = f.read()
        fingerprints = text.split("|")
        print(len(fingerprints))
        for i in range(len(fingerprints) - 1):
            print(i)
            fPrint = fingerprints[i].split(",")
            fPrint.pop()
            fingerprint = map(int,fPrint)
            printList.append(fingerprint)
        f.close()
    return printList

def get_indices_from_file(path):
    # returns the list of ending indices from a file
    with open(path, encoding="utf-8") as f:
        text = f.read()
        indices = text.split("|")
        f.close()
    return indices




