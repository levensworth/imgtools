#!/bin/zsh

echo "GOING TO RUN BLACK CODE LINTER ..."
`black ./`
echo "GOING TO RUN ISORT TO SORT IMPORTS ..."
`isort ./`
echo "DONE! NOW YOU CAN COMMIT"
