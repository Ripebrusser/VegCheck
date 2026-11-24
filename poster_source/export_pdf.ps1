# Export the poster to print-ready PDF and a full-poster PNG preview.
#
#   powershell -ExecutionPolicy Bypass -File poster_source/export_pdf.ps1
#
# Requires Microsoft PowerPoint (COM automation). The PDF keeps the slide's
# physical size (1189 x 841 mm, A0 landscape).

param(
    [string]$Pptx = "IEEE_Conference_Poster.pptx",
    [int]$PreviewWidth = 3600,
    [string]$Png = "poster_preview.png"
)

$ErrorActionPreference = "Stop"
$src = (Resolve-Path $Pptx).Path
$dir = Split-Path $src
$pdf = [System.IO.Path]::ChangeExtension($src, ".pdf")
$png = Join-Path $dir $Png

$app = New-Object -ComObject PowerPoint.Application
$pres = $app.Presentations.Open($src, $true, $false, $false)

# 32 = ppSaveAsPDF
$pres.SaveAs($pdf, 32)

$slide = $pres.Slides.Item(1)
$aspect = $pres.PageSetup.SlideHeight / $pres.PageSetup.SlideWidth
$height = [int]($PreviewWidth * $aspect)
$slide.Export($png, "PNG", $PreviewWidth, $height)

$pres.Close()
$app.Quit()

Write-Output "PDF : $pdf"
Write-Output "PNG : $png  (${PreviewWidth}x${height})"
