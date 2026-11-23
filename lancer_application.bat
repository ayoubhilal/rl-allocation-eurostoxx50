@echo off
rem Lance l'application (tableau de bord, mise a jour hebdomadaire, resultats, verifications).
rem Double-cliquer sur ce fichier ; laisser la fenetre ouverte pendant l'utilisation.
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    echo Environnement .venv introuvable : utilisation du Python du systeme.
    echo Pour l'installer, voir la section "Installation et execution" du README.
    set "PY=python"
)
"%PY%" app\server.py
if errorlevel 1 pause
