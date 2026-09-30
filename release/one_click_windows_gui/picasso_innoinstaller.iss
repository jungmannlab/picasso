; VARIANT is an optional suffix used to build alternative editions (e.g.
; the CUDA build passes /DVARIANT=-CUDA). When left undefined it defaults to
; an empty string, reproducing the standard CPU installer names exactly.
#ifndef VARIANT
  #define VARIANT ""
#endif

; DISTDIR is the PyInstaller output folder to package. The CUDA build writes to
; a separate "dist_cuda" tree (passed via /DDISTDIR=dist_cuda) so it does not
; clash with the CPU build's "dist". Defaults to "dist" for the CPU installer.
#ifndef DISTDIR
  #define DISTDIR "dist"
#endif

[Setup]
AppName=Picasso{#VARIANT}
AppPublisher=Jungmann Lab, Max Planck Institute of Biochemistry
AppVersion={#APP_VERSION}
DefaultDirName="C:\Picasso{#VARIANT}"
DefaultGroupName=Picasso{#VARIANT}
OutputBaseFilename="Picasso-Windows-64bit{#VARIANT}-{#APP_VERSION}"
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Files]
Source: "{#DISTDIR}\picasso\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs

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

[Icons]
Name: "{group}\Design"; Filename: "{app}\picassow.exe"; Parameters: "design"; IconFilename: "{app}\_internal\picasso\gui\icons\design.ico"; Components: design
Name: "{group}\Simulate"; Filename: "{app}\picassow.exe"; Parameters: "simulate"; IconFilename: "{app}\_internal\picasso\gui\icons\simulate.ico"; Components: simulate
Name: "{group}\Localize"; Filename: "{app}\picassow.exe"; Parameters: "localize"; IconFilename: "{app}\_internal\picasso\gui\icons\localize.ico"; Components: localize
Name: "{group}\Filter"; Filename: "{app}\picassow.exe"; Parameters: "filter"; IconFilename: "{app}\_internal\picasso\gui\icons\filter.ico"; Components: filter
Name: "{group}\Render"; Filename: "{app}\picassow.exe"; Parameters: "render"; IconFilename: "{app}\_internal\picasso\gui\icons\render.ico"; Components: render
Name: "{group}\Average"; Filename: "{app}\picassow.exe"; Parameters: "average"; IconFilename: "{app}\_internal\picasso\gui\icons\average.ico"; Components: average
Name: "{group}\SPINNA"; Filename: "{app}\picassow.exe"; Parameters: "spinna"; IconFilename: "{app}\_internal\picasso\gui\icons\spinna.ico"; Components: spinna
Name: "{group}\Server"; Filename: "{app}\picasso.exe"; Parameters: "server"; IconFilename: "{app}\_internal\picasso\gui\icons\server.ico"; Components: server
Name: "{group}\Nanotron"; Filename: "{app}\picassow.exe"; Parameters: "nanotron"; IconFilename: "{app}\_internal\picasso\gui\icons\nanotron.ico"; Components: nanotron


Name: "{autodesktop}\Design"; Filename: "{app}\picassow.exe"; Parameters: "design"; IconFilename: "{app}\_internal\picasso\gui\icons\design.ico"; Components: design
Name: "{autodesktop}\Simulate"; Filename: "{app}\picassow.exe"; Parameters: "simulate"; IconFilename: "{app}\_internal\picasso\gui\icons\simulate.ico"; Components: simulate
Name: "{autodesktop}\Localize"; Filename: "{app}\picassow.exe"; Parameters: "localize"; IconFilename: "{app}\_internal\picasso\gui\icons\localize.ico"; Components: localize
Name: "{autodesktop}\Filter"; Filename: "{app}\picassow.exe"; Parameters: "filter"; IconFilename: "{app}\_internal\picasso\gui\icons\filter.ico"; Components: filter
Name: "{autodesktop}\Render"; Filename: "{app}\picassow.exe"; Parameters: "render"; IconFilename: "{app}\_internal\picasso\gui\icons\render.ico"; Components: render
Name: "{autodesktop}\Average"; Filename: "{app}\picassow.exe"; Parameters: "average"; IconFilename: "{app}\_internal\picasso\gui\icons\average.ico"; Components: average
Name: "{autodesktop}\SPINNA"; Filename: "{app}\picassow.exe"; Parameters: "spinna"; IconFilename: "{app}\_internal\picasso\gui\icons\spinna.ico"; Components: spinna
Name: "{autodesktop}\Server"; Filename: "{app}\picasso.exe"; Parameters: "server"; IconFilename: "{app}\_internal\picasso\gui\icons\server.ico"; Components: server
Name: "{autodesktop}\Nanotron"; Filename: "{app}\picassow.exe"; Parameters: "nanotron"; IconFilename: "{app}\_internal\picasso\gui\icons\nanotron.ico"; Components: nanotron

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
