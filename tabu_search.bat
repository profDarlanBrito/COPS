@echo off
echo Starting script tabu search ...

cd %2
echo %cd%

set command=%1 tabu_search.py --path %3
echo %command%

%command%