[Setup]
AppName=Picasso
AppPublisher=Jungmann Lab, Max Planck Institute of Biochemistry
AppVersion=0.1
DefaultDirName={pf}\Picasso
DefaultGroupName=Picasso
OutputBaseFilename="Picasso-Installer"
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

[Files]
Source: "E:\data\schnitzbauer\picasso.pkg\dist\picasso\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Localize"; Filename: "{app}\picassow.exe"; Parameters: "localize"; IconFilename: "{app}\picasso\gui\icons\localize.ico"
Name: "{group}\Filter"; Filename: "{app}\picassow.exe"; Parameters: "filter"; IconFilename: "{app}\picasso\gui\icons\filter.ico"
Name: "{group}\Render"; Filename: "{app}\picassow.exe"; Parameters: "render"; IconFilename: "{app}\picasso\gui\icons\render.ico"

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