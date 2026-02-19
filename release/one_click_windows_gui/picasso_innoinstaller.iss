[Setup]
AppName=Picasso
AppPublisher=Jungmann Lab, Max Planck Institute of Biochemistry
AppVersion=0.9.6
DefaultDirName="C:\Picasso"
DefaultGroupName=Picasso
OutputBaseFilename="Picasso-Windows-64bit-0.9.6"
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

[Files]
Source: "dist\picasso\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs

[Types]
Name: "full"; Description: "Full installation"
Name: "custom"; Description: "Custom installation"; Flags: iscustom

[Components]
Name: "design"; Description: "Design"; Types: full custom
Name: "localize"; Description: "Localize"; Types: full custom
Name: "simulate"; Description: "Simulate"; Types: full custom
Name: "filter"; Description: "Filter"; Types: full custom
Name: "render"; Description: "Render"; Types: full custom
Name: "average"; Description: "Average"; Types: full custom
Name: "spinna"; Description: "SPINNA"; Types: full custom
Name: "server"; Description: "Server"; Types: full custom
Name: "nanotron"; Description: "Nanotron"; Types: full custom
Name: "toraw"; Description: "ToRaw"; Types: full custom

[Icons]
Name: "{group}\Design"; Filename: "{app}\picassow.exe"; Parameters: "design"; IconFilename: "{app}\_internal\picasso\gui\icons\design.ico"
Name: "{group}\Simulate"; Filename: "{app}\picassow.exe"; Parameters: "simulate"; IconFilename: "{app}\_internal\picasso\gui\icons\simulate.ico"
Name: "{group}\Localize"; Filename: "{app}\picassow.exe"; Parameters: "localize"; IconFilename: "{app}\_internal\picasso\gui\icons\localize.ico"
Name: "{group}\Filter"; Filename: "{app}\picassow.exe"; Parameters: "filter"; IconFilename: "{app}\_internal\picasso\gui\icons\filter.ico"
Name: "{group}\Render"; Filename: "{app}\picassow.exe"; Parameters: "render"; IconFilename: "{app}\_internal\picasso\gui\icons\render.ico"
Name: "{group}\Average"; Filename: "{app}\picassow.exe"; Parameters: "average"; IconFilename: "{app}\_internal\picasso\gui\icons\average.ico"
Name: "{group}\SPINNA"; Filename: "{app}\picassow.exe"; Parameters: "spinna"; IconFilename: "{app}\_internal\picasso\gui\icons\spinna.ico"
Name: "{group}\Server"; Filename: "{app}\picasso.exe"; Parameters: "server"; IconFilename: "{app}\_internal\picasso\gui\icons\server.ico"
Name: "{group}\Nanotron"; Filename: "{app}\picassow.exe"; Parameters: "nanotron"; IconFilename: "{app}\_internal\picasso\gui\icons\nanotron.ico"
Name: "{group}\ToRaw"; Filename: "{app}\picassow.exe"; Parameters: "toraw"; IconFilename: "{app}\_internal\picasso\gui\icons\toraw.ico"


Name: "{autodesktop}\Design"; Filename: "{app}\picassow.exe"; Parameters: "design"; IconFilename: "{app}\_internal\picasso\gui\icons\design.ico"
Name: "{autodesktop}\Simulate"; Filename: "{app}\picassow.exe"; Parameters: "simulate"; IconFilename: "{app}\_internal\picasso\gui\icons\simulate.ico"
Name: "{autodesktop}\Localize"; Filename: "{app}\picassow.exe"; Parameters: "localize"; IconFilename: "{app}\_internal\picasso\gui\icons\localize.ico"
Name: "{autodesktop}\Filter"; Filename: "{app}\picassow.exe"; Parameters: "filter"; IconFilename: "{app}\_internal\picasso\gui\icons\filter.ico"
Name: "{autodesktop}\Render"; Filename: "{app}\picassow.exe"; Parameters: "render"; IconFilename: "{app}\_internal\picasso\gui\icons\render.ico"
Name: "{autodesktop}\Average"; Filename: "{app}\picassow.exe"; Parameters: "average"; IconFilename: "{app}\_internal\picasso\gui\icons\average.ico"
Name: "{autodesktop}\SPINNA"; Filename: "{app}\picassow.exe"; Parameters: "spinna"; IconFilename: "{app}\_internal\picasso\gui\icons\spinna.ico"
Name: "{autodesktop}\Server"; Filename: "{app}\picasso.exe"; Parameters: "server"; IconFilename: "{app}\_internal\picasso\gui\icons\server.ico"
Name: "{autodesktop}\Nanotron"; Filename: "{app}\picassow.exe"; Parameters: "nanotron"; IconFilename: "{app}\_internal\picasso\gui\icons\nanotron.ico"
Name: "{autodesktop}\ToRaw"; Filename: "{app}\picassow.exe"; Parameters: "toraw"; IconFilename: "{app}\_internal\picasso\gui\icons\toraw.ico"

[Registry]
Root: HKLM; Subkey: "SYSTEM\CurrentControlSet\Control\Session Manager\Environment"; \
    ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{app}"; \
    Check: NeedsAddPath('{app}')

[Code]
function NeedsAddPath(Param: string): boolean;
var
  OrigPath: string;
  ParamExpanded: string;
begin
  //expand the setup constants like {app} from Param
  ParamExpanded := ExpandConstant(Param);
  if not RegQueryStringValue(HKEY_LOCAL_MACHINE,
    'SYSTEM\CurrentControlSet\Control\Session Manager\Environment',
    'Path', OrigPath)
  then begin
    Result := True;
    exit;
  end;
  // look for the path with leading and trailing semicolon and with or without \ ending
  // Pos() returns 0 if not found
  Result := Pos(';' + UpperCase(ParamExpanded) + ';', ';' + UpperCase(OrigPath) + ';') = 0;
  if Result = True then
     Result := Pos(';' + UpperCase(ParamExpanded) + '\;', ';' + UpperCase(OrigPath) + ';') = 0;
end;
