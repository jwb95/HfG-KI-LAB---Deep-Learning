Um eine Conda Environment zu erstellen, die identisch zu den Python-/Tensorflow-Versionen auf Colab ist, kann die "colab_env.yml" genutzt werden.

1. Speichere colab_env.yml auf bspw. deinem Desktop.
2. Öfnne die Anaconda Prompt.
3. Navigiere zum Desktop:
```
cd Desktop
```
Der Default-Name der Environement ist 'tf2'. Um diese zu ändern kann man die erste Line der 'colab_env.yml' anpassen.

4. Dann um die Environment zu erstellen:
```
conda env create -f colab_env.yml
```


https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
