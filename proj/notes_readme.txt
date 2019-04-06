Reuters-21578

- 22 sgm files
-- reut2-000.sgm to -020.sgm - 1000 docs
-- last reut2-021.sgm - 578 docs

- sgml format
- sgml tags divide each file and each doc into sections

- each .sgm file begins with doctype declaration

- after this, individual Reuters articles are marked up with SGML tags
--> article starts with open tag <REUTERS TOPICS=?? LEWISSPLIT=?? CGISPLIT=?? OLDID=?? NEWID=??>

--> article ends with close tag </REUTERS>
** <REUTERS> and <\REUTERS> tags delimit documents within a file.

> attributes of REUTERS tag:
1. TOPICS: 
indicate whether or not the document had TOPICS categories *in the raw Reuters-22173 dataset*
helps in defining training set splits
**** does **NOT** indicate anything about whether or not the Reuters-21578 document has any TOPICS categories *****
 - Yes: *in the original data* there was at least one entry in the TOPICS fields
 - No:no entry in topics fields
 - Bypass: in the original data* the story was marked with the string "bypass"

Negative Examples
 -> the TOPICS="YES" stories with no topics can reasonably be considered negative examples for all 135 valid TOPICS categories

 * TOPICS="NO" stories appear to be concentrated in certain time intrvals. some parts of the data set were simply not indexed, or
not indexed for some categories or category sets

2. LEWISSPLIT:
For experiments in LEWIS91d (Chapters 9 and 10), LEWIS92b, LEWIS92e, and LEWIS94b
 - Training: indicates it was used in the training set
 - Test: it was used in the test set
 - Not used

3. CGISPLIT:
For the experiments reported in HAYES89 and HAYES90b:
 - Training Set document was in the training  set
 - Published Testset: document was in the test set

4. OLDID: id in r-22173 collection

5. NEWID: id in r-21578 collection; chronological

tags delimiting elements within a doc:
(imp ones)
--> <TOPICS>, </TOPICS> [ONCE, SAMELINE]: Encloses the list of TOPICS categories, if any, for the document. If TOPICS categories are present, each will be delimited by the tags <D> and </D>.

--> similarly <PLACES>, <PEOPLE>, <ORGS>, <EXCHANGES>, <COMPANIES>

--> Text: delimit all the textual material of each story between a pair of these tags.
> attributes of TEXT tag:
1. TYPE
 - Norm: default
 		 text of story had a normal structure
 - Brief: for short stories
 - Unproc: for unusual stories
optional:
2. [author], [dateline], [title], [body]


______-----------_______________________________

Categories
the categories are five different sets of content related categories. 
              Number of    Number of Categories   Number of Categories 
Category Set  Categories     w/ 1+ Occurrences      w/ 20+ Occurrences  
************  **********   ********************   ******************** 
EXCHANGES        39                32                       7
ORGS             56                32                       9
PEOPLE          267               114                      15
PLACES          175               147                      60
TOPICS          135               120                      57


TOPICS categories are economic subject categories.  Examples include "coconut", "gold", "inventories", and "money-supply".  This set of categories is the one that has been used in almost all previous research with the Reuters data.

The EXCHANGES, ORGS, PEOPLE, and PLACES categories correspond to named entities of the specified type.

(all-exchanges-strings.lc.txt, all-orgs-strings.lc.txt, all-people-strings.lc.txt, all-places-strings.lc.txt, and all-topics-strings.lc.txt) which list the names of *all* legal categories in each set

cat-descriptions_120396.txt gives some additional information on the category sets.
