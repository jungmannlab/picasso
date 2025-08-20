[Setup]
AppName=Picasso
AppPublisher=Jungmann Lab, Max Planck Institute of Biochemistry

AppVersion=0.8.3
DefaultDirName={commonpf}\Picasso
DefaultGroupName=Picasso
OutputBaseFilename="Picasso-Windows-64bit-0.8.3"
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

[Files]
Source: "dist\picasso\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Design"; Filename: "{app}\picassow.exe"; Parameters: "design"; IconFilename: "{app}\picasso\gui\icons\design.ico"
Name: "{group}\Simulate"; Filename: "{app}\picassow.exe"; Parameters: "simulate"; IconFilename: "{app}\picasso\gui\icons\simulate.ico"
Name: "{group}\Localize"; Filename: "{app}\picassow.exe"; Parameters: "localize"; IconFilename: "{app}\picasso\gui\icons\localize.ico"
Name: "{group}\Filter"; Filename: "{app}\picassow.exe"; Parameters: "filter"; IconFilename: "{app}\picasso\gui\icons\filter.ico"
Name: "{group}\Render"; Filename: "{app}\picassow.exe"; Parameters: "render"; IconFilename: "{app}\picasso\gui\icons\render.ico"
Name: "{group}\Average"; Filename: "{app}\picassow.exe"; Parameters: "average"; IconFilename: "{app}\picasso\gui\icons\average.ico"
Name: "{group}\Server"; Filename: "{app}\picasso.exe"; Parameters: "server"; IconFilename: "{app}\picasso\gui\icons\server.ico"
Name: "{group}\SPINNA"; Filename: "{app}\picasso.exe"; Parameters: "spinna"; IconFilename: "{app}\picasso\gui\icons\spinna.ico"

Name: "{autodesktop}\Design"; Filename: "{app}\picassow.exe"; Parameters: "design"; IconFilename: "{app}\picasso\gui\icons\design.ico"
Name: "{autodesktop}\Simulate"; Filename: "{app}\picassow.exe"; Parameters: "simulate"; IconFilename: "{app}\picasso\gui\icons\simulate.ico"
Name: "{autodesktop}\Localize"; Filename: "{app}\picassow.exe"; Parameters: "localize"; IconFilename: "{app}\picasso\gui\icons\localize.ico"
Name: "{autodesktop}\Filter"; Filename: "{app}\picassow.exe"; Parameters: "filter"; IconFilename: "{app}\picasso\gui\icons\filter.ico"
Name: "{autodesktop}\Render"; Filename: "{app}\picassow.exe"; Parameters: "render"; IconFilename: "{app}\picasso\gui\icons\render.ico"
Name: "{autodesktop}\Average"; Filename: "{app}\picassow.exe"; Parameters: "average"; IconFilename: "{app}\picasso\gui\icons\average.ico"
Name: "{autodesktop}\Server"; Filename: "{app}\picasso.exe"; Parameters: "server"; IconFilename: "{app}\picasso\gui\icons\server.ico"
Name: "{autodesktop}\SPINNA"; Filename: "{app}\picasso.exe"; Parameters: "spinna"; IconFilename: "{app}\picasso\gui\icons\spinna.ico"

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