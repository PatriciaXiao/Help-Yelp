find . -size +50M
echo "the above files are too large to be uploaded onto Github"
echo "keep them local instead"
# find . -size +50M | cat >> .gitignore
find . -size +50M | cat > .gitignore

echo "cleaning the format to a universally appied version"
python clean_format.py
