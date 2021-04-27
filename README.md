# Index Funds and Corporate Governance

## Evidence from State Street Global Advisors

Author: Bing Xu

Research essay written as the major part of my Honours in Finance at the University of Melbourne. 
Final mark was a H1. 

Directory is as follows:
- /src contains the python scripts used. The entrypoint is analysis.py
- /data contains the data used - this data is not the original data but the data following processing and cleaning
- /output contains the output from the regressions/statistical analysis.

## Abstract
The Big Three index fund managers, BlackRock, Vanguard and State Street Global
Advisors, continue to wield an increasingly large share of global equity. The question
remains as to what extent they are able to use their large stakes to affect governance. I
shed light on the mechanisms of proxy voting and behind-the-scenes engagements they
use to exert institutional voice. My findings demonstrate a large degree of heterogeneity
exists between the Big Three in their proxy voting. Disagreement largely occurs with
respect to director elections, pay approval and other shareholder proposals. Vanguard
and BlackRock cast a significantly higher proportion of their votes in favour of
management relative to State Street. This is consistent across proposals of all types.
Using publicly available engagement data from State Street, I demonstrate private
engagement informs proxy voting. Engagements appear to act as a means of obtaining
private firm-specific information, and I document significant differences between State
Street’s voting behaviour at shareholder meetings with and without engagement. State
Street’s votes appear to be motivated by fundamentals, and there is strong evidence to
suggest they actively monitor portfolio firms.

## Getting started - running this code
1. Clone this repository
```bash
git clone https://github.com/bingx1/indexfunds.git
```
2. Move into the directory
```bash
cd indexfunds/
```
3. Create a new virtual environment
```bash
python3 -m  venv venv
```
4. Activate the virtual environment
```bash
source venv/bin/activate
```
5. Install the project dependencies
```bash
pip install -r requirements.txt
```
6. Run the scripts
```bash
cd src && python3 analysis.py
```