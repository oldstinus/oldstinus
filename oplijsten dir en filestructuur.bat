@echo off
rem Haal de huidige directory op waar het batchbestand wordt uitgevoerd
set "currentDir=%~dp0"

rem Haal de naam van de huidige directory op zonder de laatste backslash
set "currentDir=%currentDir:~0,-1%"

rem Stel de bestandsnamen in
set tempFile=temp_structure.txt
set csharpFile=DirectoryStructure.cs

rem Verwijder eventuele bestaande tijdelijke en C#-bestanden
if exist %tempFile% del %tempFile%
if exist %csharpFile% del %csharpFile%

rem Schrijf de directorystructuur naar een tijdelijk tekstbestand
echo Schrijven van directorystructuur van de huidige map naar %tempFile%...
tree "%currentDir%" /F /A > %tempFile%

rem Maak het C#-bestand aan en schrijf de inhoud weg
echo using System; > %csharpFile%
echo using System.IO; >> %csharpFile%
echo >> %csharpFile%
echo namespace DirectoryStructureApp >> %csharpFile%
echo { >> %csharpFile%
echo     class Program >> %csharpFile%
echo     { >> %csharpFile%
echo         static void Main(string[] args) >> %csharpFile%
echo         { >> %csharpFile%
echo             string[] structure = new string[] >> %csharpFile%
echo             { >> %csharpFile%

rem Lees de inhoud van het tijdelijke bestand en zet het om in een C#-array
for /f "delims=" %%i in (%tempFile%) do (
    echo                 "%%i", >> %csharpFile%
)

rem Sluit de C#-array en de code af
echo             }; >> %csharpFile%
echo             File.WriteAllLines("directory_structure.txt", structure); >> %csharpFile%
echo             Console.WriteLine("Directorystructuur is weggeschreven naar directory_structure.txt"); >> %csharpFile%
echo         } >> %csharpFile%
echo     } >> %csharpFile%
echo } >> %csharpFile%

rem Verwijder het tijdelijke bestand
del %tempFile%

echo C#-bestand %csharpFile% is aangemaakt met de directorystructuur van %currentDir%.
pause
