# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# TODO: Address all TODOs and remove all explanatory comments
"""TODO: Add a description here."""


import csv
import json
import os
import enum
import datasets
from pathlib import Path  

logger = datasets.logging.get_logger(__name__)

# TODO: Add BibTeX citation
_CITATION = """\
@InProceedings{huggingface:dataset,
title = {A great new dataset},
author={huggingface, Inc.
},
year={2022}
}
"""

# TODO: Add description of the dataset here
_DESCRIPTION = """\
This new dataset is designed to solve this great NLP task and is crafted with a lot of care.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = "ToDo"

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = "ToDo"

# TODO: Add link to the official dataset URLs here

# use `path_config.json` for filtered experiments
# with open('path_config.json') as f:
#     path_config = json.load(f)

# use `path_config_full_splits.json` for gathering statistics
with open('path_config_full_splits.json') as f:
    path_config = json.load(f)


_URL = Path(path_config['path_to_splits'])
_URLS = {
    "train": _URL / path_config['train_split'],
    "validation": _URL / path_config['valid_split'], # dev items with no mC4-overlapping articles
    "test": _URL / path_config['test_split'], # test items with no mC4-overlapping articles
}

print(f"Dataset paths train: {_URLS['train']}, validation: {_URLS['validation']}, test: {_URLS['test']}")

class TwentyMinTasks(enum.Enum): 
    TITLE="title_generation",
    LEAD="lead_generation",
    SUMMARY="summary_generation",
    CAPTION="caption_generation",
    SUMMARY_AND_CAPTION="summary_and_caption",
    ALL="all_data",
    READING_TIME="reading_time_prediction",
    CATEGORY="category_prediction",
    def __init__(self, taskName):
        self.task_name = taskName

    
class TwentyMinData(enum.Enum): 
    ID="id",
    DATE="date",
    DATE_CREATED="dateCreated",
    DATE_PUBLISHED="datePublished",
    DATE_UPDATED="dateUpdated",
    READING_TIME="readingTime",
    AUTHOR="author",
    CATEGORY="category",
    URL="url",
    TITLE_HEADER="titleHeader",
    TITLE="title",
    LEAD="lead",
    CAPTION="pictureText",
    SUMMARY="summary",
    CONTENT="content"
    def __init__(self, dataName):
        self.data_name = dataName
    
 
class TwentyMinConfig(datasets.BuilderConfig):
    """BuilderConfig for TwentyMin"""


    def __init__(self, 
                 skipEmpty,
                 **kwargs):
        """
        Args:
            **kwargs: keyword arguments forwarded to super.
        """
        super(TwentyMinConfig, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)
        self.skipEmpty = skipEmpty


class TwentyMin(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        TwentyMinConfig(
            name=TwentyMinTasks.LEAD.task_name,
            description="Predict leads for the article",
            skipEmpty=True
            ),
        TwentyMinConfig(
            name=TwentyMinTasks.SUMMARY.task_name,
            description="Predict summary for the article",
            skipEmpty=True
            ),
        TwentyMinConfig(
            name=TwentyMinTasks.CAPTION.task_name,
            description="Predict picture text for the article",
            skipEmpty=True
            ),
        TwentyMinConfig(
            name=TwentyMinTasks.SUMMARY_AND_CAPTION.task_name,
            description="Predict summary and picture text for the article",
            skipEmpty=True
            ),
        TwentyMinConfig(
            name=TwentyMinTasks.ALL.task_name,
            description="Return all data",
            skipEmpty=False
            ),
        TwentyMinConfig(
            name=TwentyMinTasks.TITLE.task_name,
            description="Predict title for the article",
            skipEmpty=True
            ),
        TwentyMinConfig(
            name=TwentyMinTasks.READING_TIME.task_name,
            description="Predict reading time for the article",
            skipEmpty=True
            ),
        TwentyMinConfig(
            name=TwentyMinTasks.CATEGORY.task_name,
            description="Predict reading time for the article",
            skipEmpty=True
            ),
    ]
    
    def _info(self):
        if self.config.name == TwentyMinTasks.ALL.task_name:
            return datasets.DatasetInfo(
                description=_DESCRIPTION,
                features=datasets.Features(
                    {
                    TwentyMinData.ID.data_name: datasets.Value("string"),
                    TwentyMinData.DATE.data_name: datasets.Value("string"),
                    TwentyMinData.DATE_CREATED.data_name: datasets.Value("string"),
                    TwentyMinData.DATE_PUBLISHED.data_name: datasets.Value("string"),
                    TwentyMinData.DATE_UPDATED.data_name: datasets.Value("string"),
                    TwentyMinData.READING_TIME.data_name:  datasets.Value("string"),
                    TwentyMinData.AUTHOR.data_name:  datasets.Value("string"),
                    TwentyMinData.CATEGORY.data_name:  datasets.features.Sequence(datasets.Value("string")),
                    TwentyMinData.URL.data_name:  datasets.Value("string"),
                    TwentyMinData.TITLE_HEADER.data_name:  datasets.Value("string"),
                    TwentyMinData.TITLE.data_name:  datasets.Value("string"),
                    TwentyMinData.LEAD.data_name:  datasets.Value("string"),
                    TwentyMinData.CAPTION.data_name: datasets.features.Sequence(datasets.Value("string")),
                    TwentyMinData.SUMMARY.data_name: datasets.features.Sequence(
                            {
                                "summary_index": datasets.Value("string"),
                                "summary_text": datasets.Value("string"),
                            }
                    ),
                    TwentyMinData.CONTENT.data_name: datasets.features.Sequence(
                            {
                                "content_index": datasets.Value("string"),
                                "content_text": datasets.Value("string"),
                            }
                    )               
                    
                    }
                ),  
                homepage=_HOMEPAGE,
                license=_LICENSE,
                citation=_CITATION,
            )
        elif self.config.name == TwentyMinTasks.LEAD.task_name:
            return datasets.DatasetInfo(
                description=_DESCRIPTION,
                features=datasets.Features(
                    {
                    TwentyMinData.ID.data_name: datasets.Value("string"),
                    TwentyMinData.LEAD.data_name: datasets.Value("string"),
                    TwentyMinData.CONTENT.data_name: datasets.Value("string")
                    }
                ),  
                homepage=_HOMEPAGE,
                license=_LICENSE,
                citation=_CITATION,
            )
        elif self.config.name == TwentyMinTasks.SUMMARY.task_name:
            return datasets.DatasetInfo(
                description=_DESCRIPTION,
                features=datasets.Features(
                    {
                    TwentyMinData.ID.data_name: datasets.Value("string"),
                    TwentyMinData.SUMMARY.data_name: datasets.features.Sequence(datasets.Value("string")),
                    TwentyMinData.CONTENT.data_name: datasets.Value("string")
                    }
                ),  
                homepage=_HOMEPAGE,
                license=_LICENSE,
                citation=_CITATION,
            )
        elif self.config.name == TwentyMinTasks.CAPTION.task_name:
            return datasets.DatasetInfo(
                description=_DESCRIPTION,
                features=datasets.Features(
                    {
                    TwentyMinData.ID.data_name: datasets.Value("string"),
                    TwentyMinData.CAPTION.data_name: datasets.features.Sequence(datasets.Value("string")),
                    TwentyMinData.CONTENT.data_name: datasets.Value("string")
                    }
                ),  
                homepage=_HOMEPAGE,
                license=_LICENSE,
                citation=_CITATION,
            )
        elif self.config.name == TwentyMinTasks.SUMMARY_AND_CAPTION.task_name:
            return datasets.DatasetInfo(
                description=_DESCRIPTION,
                features=datasets.Features(
                    {
                    TwentyMinData.ID.data_name: datasets.Value("string"),
                    TwentyMinData.SUMMARY.data_name: datasets.features.Sequence(datasets.Value("string")),
                    TwentyMinData.CAPTION.data_name: datasets.features.Sequence(datasets.Value("string")),
                    TwentyMinData.CONTENT.data_name: datasets.Value("string")
                    }
                ),  
                homepage=_HOMEPAGE,
                license=_LICENSE,
                citation=_CITATION,
            )
        elif self.config.name == TwentyMinTasks.TITLE.task_name:
            return datasets.DatasetInfo(
                description=_DESCRIPTION,
                features=datasets.Features(
                    {
                    TwentyMinData.ID.data_name: datasets.Value("string"),
                    TwentyMinData.TITLE.data_name: datasets.Value("string"),
                    TwentyMinData.TITLE_HEADER.data_name: datasets.Value("string"),
                    TwentyMinData.CONTENT.data_name: datasets.Value("string")
                    }
                ),  
                homepage=_HOMEPAGE,
                license=_LICENSE,
                citation=_CITATION,
            )
        elif self.config.name == TwentyMinTasks.READING_TIME.task_name:
            return datasets.DatasetInfo(
                description=_DESCRIPTION,
                features=datasets.Features(
                    {
                    TwentyMinData.ID.data_name: datasets.Value("string"),
                    TwentyMinData.READING_TIME.data_name: datasets.Value("string"),
                    TwentyMinData.CONTENT.data_name: datasets.Value("string")
                    }
                ),  
                homepage=_HOMEPAGE,
                license=_LICENSE,
                citation=_CITATION,
            )
        elif self.config.name == TwentyMinTasks.CATEGORY.task_name:
            return datasets.DatasetInfo(
                description=_DESCRIPTION,
                features=datasets.Features(
                    {
                    TwentyMinData.ID.data_name: datasets.Value("string"),
                    TwentyMinData.CATEGORY.data_name: datasets.features.Sequence(datasets.Value("string")),
                    TwentyMinData.CONTENT.data_name: datasets.Value("string")
                    }
                ),  
                homepage=_HOMEPAGE,
                license=_LICENSE,
                citation=_CITATION,
            )


    def _getDataDir(self, data_dir):
        filepaths = [
            os.path.join(data_dir, txt_file_name)
            for txt_file_name in sorted(os.listdir(data_dir))
            if txt_file_name.endswith("json")
        ]
        return filepaths

    def _getGenerator(self, dataDir, typeName, splitName):
        return datasets.SplitGenerator(name=typeName, gen_kwargs={"filepaths": self._getDataDir(dataDir), "split": splitName})

    def _split_generators(self, dl_manager):
        data_sets = []
        # breakpoint()
        if (self.config.data_dir and 'train' in  self.config.data_dir) or 'train' in _URLS:
             if (self.config.data_dir and 'train' in  self.config.data_dir):
                data_dir =  self.config.data_dir['train']
             elif 'train' in _URLS:
                data_dir =  _URLS['train']
             data_sets.append(self._getGenerator(data_dir, datasets.Split.TRAIN, "train"))
        if (self.config.data_dir and 'validation' in  self.config.data_dir) or 'validation' in _URLS:
            if (self.config.data_dir and 'validation' in  self.config.data_dir):
                data_dir =  self.config.data_dir['validation']
            elif 'validation' in _URLS:
                data_dir =  _URLS['validation']
            data_sets.append(self._getGenerator(data_dir, datasets.Split.VALIDATION, "dev"))
        if (self.config.data_dir and 'test' in  self.config.data_dir) or 'test' in _URLS: 
            if (self.config.data_dir and 'test' in  self.config.data_dir):
                data_dir =  self.config.data_dir['test']
            elif 'test' in _URLS:
                data_dir =  _URLS['test']
            data_sets.append(self._getGenerator(data_dir, datasets.Split.TEST, "test"))
        return data_sets

    def _extract_data(self, data, key, isList=False):
        item = None
        if key in data:
            item = data[key]
        if isList: 
            if isinstance(item, list):
                return item
            else:
                return [item]
        else:
            return item
        return ""
    
    def _extract_content(self, data, keyName, outPutKeyNames, shouldConcatenate=False):
        out = []
        if not shouldConcatenate:
            for key in data[keyName]:
                if len(outPutKeyNames)>1:
                    out.append( { outPutKeyNames[0]: key, outPutKeyNames[1]: data[keyName][key] })
                elif len(outPutKeyNames)==0:
                    out.append(data[keyName][key])
            return out
        else:
            out_txt = ""
            for key in data[keyName]:
                key_value = data[keyName][key]
                if isinstance(key_value, list):
                    for item in key_value:
                        out_txt += item + " "
                else:                        
                    out_txt += data[keyName][key] + " "
            out.append(out_txt)
            return ' '.join(map(str, out))
    
    def _should_skip(self, value):
        return value is None or value == "" or value == "None"
    
    def _check_to_skip(self, shouldSkip, dictVal):
        if shouldSkip:
            for key in dictVal.keys():
                if isinstance(dictVal[key], list):
                    if len(dictVal[key]) ==0:
                        return shouldSkip
                    for item in dictVal[key]:
                        if self._should_skip(item):
                            return shouldSkip
                elif self._should_skip(dictVal[key]):
                    return shouldSkip
            shouldSkip=False
        return shouldSkip
            

    def _generate_examples(self, filepaths, split):
        for file_path in filepaths:
            data = {}
            try:
                with open(file_path, encoding='utf-8') as fh:
                    data = json.load(fh) 
            except UnicodeDecodeError:
                logger.error("Encoding error ", file_path)
            if self.config.name == TwentyMinTasks.ALL.task_name:
                yield Path(file_path).name, {
                    TwentyMinData.ID.data_name: self._extract_data(data, TwentyMinData.ID.data_name),
                    TwentyMinData.DATE.data_name: self._extract_data(data, TwentyMinData.DATE.data_name),
                    TwentyMinData.DATE_CREATED.data_name: self._extract_data(data, TwentyMinData.DATE_CREATED.data_name),
                    TwentyMinData.DATE_PUBLISHED.data_name: self._extract_data(data, TwentyMinData.DATE_PUBLISHED.data_name),
                    TwentyMinData.DATE_UPDATED.data_name: self._extract_data(data, TwentyMinData.DATE_UPDATED.data_name),
                    TwentyMinData.READING_TIME.data_name: self._extract_data(data, TwentyMinData.READING_TIME.data_name),
                    TwentyMinData.AUTHOR.data_name: self._extract_data(data, TwentyMinData.AUTHOR.data_name),
                    TwentyMinData.CATEGORY.data_name: self._extract_data(data, TwentyMinData.CATEGORY.data_name, isList=True),
                    TwentyMinData.URL.data_name: self._extract_data(data, TwentyMinData.URL.data_name),
                    TwentyMinData.TITLE_HEADER.data_name: self._extract_data(data, TwentyMinData.TITLE_HEADER.data_name),
                    TwentyMinData.TITLE.data_name: self._extract_data(data, TwentyMinData.TITLE.data_name),
                    TwentyMinData.LEAD.data_name: self._extract_data(data, TwentyMinData.LEAD.data_name), 
                    TwentyMinData.CAPTION.data_name: self._extract_data(data, TwentyMinData.CAPTION.data_name, isList=True),
                    TwentyMinData.SUMMARY.data_name: self._extract_content(data, TwentyMinData.SUMMARY.data_name, ["summary_index", "summary_text"]),
                    TwentyMinData.CONTENT.data_name : self._extract_content(data, TwentyMinData.CONTENT.data_name, ["content_index", "content_text"])      
                }
            elif self.config.name == TwentyMinTasks.LEAD.task_name:
                dic = {
                    TwentyMinData.ID.data_name: self._extract_data(data, TwentyMinData.ID.data_name),
                    TwentyMinData.LEAD.data_name: self._extract_data(data, TwentyMinData.LEAD.data_name), 
                    TwentyMinData.CONTENT.data_name : self._extract_content(data, TwentyMinData.CONTENT.data_name, ["content_index", "content_text"], shouldConcatenate=True)      
                }
                if not self._check_to_skip(self.config.skipEmpty, dic):
                    yield Path(file_path).name, dic
            elif self.config.name == TwentyMinTasks.SUMMARY.task_name:
                dic = {
                    TwentyMinData.ID.data_name: self._extract_data(data, "id"),
                    TwentyMinData.SUMMARY.data_name: self._extract_content(data, TwentyMinData.SUMMARY.data_name, []),
                    TwentyMinData.CONTENT.data_name : self._extract_content(data, TwentyMinData.CONTENT.data_name, ["content_index", "content_text"], shouldConcatenate=True)      
                }
                if not self._check_to_skip(self.config.skipEmpty, dic):
                    yield Path(file_path).name, dic
            elif self.config.name == TwentyMinTasks.CAPTION.task_name:
                dic = {
                    TwentyMinData.ID.data_name: self._extract_data(data, "id"),
                    TwentyMinData.CAPTION.data_name: self._extract_data(data, TwentyMinData.CAPTION.data_name, isList=True),
                    TwentyMinData.CONTENT.data_name : self._extract_content(data, TwentyMinData.CONTENT.data_name, ["content_index", "content_text"], shouldConcatenate=True)      
                }
                if not self._check_to_skip(self.config.skipEmpty, dic):
                    yield Path(file_path).name, dic
            elif self.config.name == TwentyMinTasks.SUMMARY_AND_CAPTION.task_name:
                dic = {
                    TwentyMinData.ID.data_name: self._extract_data(data, "id"),
                    TwentyMinData.SUMMARY.data_name: self._extract_content(data, TwentyMinData.SUMMARY.data_name, []),
                    TwentyMinData.CAPTION.data_name: self._extract_data(data, TwentyMinData.CAPTION.data_name, isList=True),
                    TwentyMinData.CONTENT.data_name : self._extract_content(data, TwentyMinData.CONTENT.data_name, ["content_index", "content_text"], shouldConcatenate=True)      
                }
                if not self._check_to_skip(self.config.skipEmpty, dic):
                    yield Path(file_path).name, dic
                
            elif self.config.name == TwentyMinTasks.TITLE.task_name:
                dic = {                
                    TwentyMinData.ID.data_name: self._extract_data(data, "id"),
                    TwentyMinData.TITLE.data_name: self._extract_data(data, TwentyMinData.TITLE.data_name), 
                    TwentyMinData.TITLE_HEADER.data_name: self._extract_data(data, TwentyMinData.TITLE_HEADER.data_name), 
                    TwentyMinData.CONTENT.data_name : self._extract_content(data, TwentyMinData.CONTENT.data_name, ["content_index", "content_text"], shouldConcatenate=True) 
                }
                if not self._check_to_skip(self.config.skipEmpty, dic):               
                    yield Path(file_path).name, dic
            elif self.config.name == TwentyMinTasks.READING_TIME.task_name:
                dic = {
                    TwentyMinData.ID.data_name: self._extract_data(data, "id"),
                    TwentyMinData.READING_TIME.data_name: self._extract_data(data, TwentyMinData.READING_TIME.data_name), 
                    TwentyMinData.CONTENT.data_name : self._extract_content(data, TwentyMinData.CONTENT.data_name, ["content_index", "content_text"], shouldConcatenate=True) 
                }
                if not self._check_to_skip(self.config.skipEmpty, dic):
                    yield Path(file_path).name, dic
            elif self.config.name == TwentyMinTasks.CATEGORY.task_name:
                dic = {                
                    TwentyMinData.ID.data_name: self._extract_data(data, "id"),
                    TwentyMinData.CATEGORY.data_name: self._extract_data(data, TwentyMinData.CATEGORY.data_name, isList=True), 
                    TwentyMinData.CONTENT.data_name : self._extract_content(data, TwentyMinData.CONTENT.data_name, ["content_index", "content_text"], shouldConcatenate=True)
                }
                if not self._check_to_skip(self.config.skipEmpty, dic):
                    yield Path(file_path).name, dic
                    

