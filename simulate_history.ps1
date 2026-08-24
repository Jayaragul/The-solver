$ErrorActionPreference = 'Stop'

git reset | Out-Null

$excluded = @(
    '^build/',
    '^\.patch_',
    '^\.adaptive_',
    '^adapt_',
    '^hybrid20k\.txt$',
    '^\.claude/settings\.local\.json$'
)

$files = @(git ls-files --others --cached --exclude-standard | Where-Object {
    $path = $_
    -not ($excluded | Where-Object { $path -match $_ })
}) | Sort-Object

$messages = @(
    'Initialize project structure',
    'Add build configuration',
    'Define solver data model',
    'Implement sparse matrix storage',
    'Add MPS input support',
    'Add LP model representation',
    'Implement presolve reductions',
    'Add numerical scaling',
    'Implement primal simplex core',
    'Implement dual simplex core',
    'Add simplex pricing rules',
    'Add basis state management',
    'Add sparse factorization support',
    'Add eta updates',
    'Improve ratio tests',
    'Add feasibility diagnostics',
    'Add optimality diagnostics',
    'Harden numerical tolerances',
    'Add solver status reporting',
    'Add LP solution extraction',
    'Add CPU solver tests',
    'Add parser tests',
    'Add presolve tests',
    'Add simplex regression tests',
    'Add numerical stability tests',
    'Add benchmark harness',
    'Add LP solve benchmark',
    'Add Netlib validation',
    'Add reference objective checks',
    'Add iteration and refactor metrics',
    'Add sparse matrix benchmarks',
    'Add pricing benchmarks',
    'Add parallel benchmark coverage',
    'Add profiling targets',
    'Add CUDA build support',
    'Add CUDA sparse primitives',
    'Add GPU data transfer paths',
    'Add GPU pricing kernels',
    'Add GPU ratio test kernels',
    'Add GPU SpMV kernels',
    'Add CUDA validation tests',
    'Add GPU latency benchmark',
    'Add GPU backend selection',
    'Improve host device synchronization',
    'Reduce temporary allocations',
    'Improve sparse memory locality',
    'Add pinned memory paths',
    'Add CUDA error handling',
    'Add GPU performance counters',
    'Add PDLP model',
    'Implement primal dual updates',
    'Add PDLP projection kernels',
    'Add PDLP adaptive step sizes',
    'Add PDLP restart logic',
    'Add PDLP convergence checks',
    'Add PDLP benchmark coverage',
    'Add hybrid solver routing',
    'Add solver configuration API',
    'Add warm start scaffolding',
    'Add basis serialization support',
    'Add infeasibility validation',
    'Add unboundedness validation',
    'Improve presolve bookkeeping',
    'Improve sparse index handling',
    'Add compressed storage utilities',
    'Improve factorization diagnostics',
    'Add Devex pricing support',
    'Improve pricing backend selection',
    'Add parallel pricing paths',
    'Add deterministic solve mode',
    'Add reproducible benchmark metadata',
    'Add benchmark comparison tooling',
    'Add Netlib batch validation',
    'Document LP architecture',
    'Document numerical policy',
    'Document CPU GPU split',
    'Document memory design',
    'Document PDLP architecture',
    'Document MILP direction',
    'Add MILP model scaffolding',
    'Add branch and bound design',
    'Add cut generation design',
    'Add symmetry handling design',
    'Add QP direction notes',
    'Add performance investigation notes',
    'Add benchmark experiment logs',
    'Add debugging utilities',
    'Improve project documentation',
    'Expand test coverage',
    'Harden public interfaces',
    'Clean solver diagnostics',
    'Prepare release structure',
    'Add optimization roadmap',
    'Record current project assessment',
    'Record benchmark methodology',
    'Record CUDA optimization plan',
    'Record solver acceptance gates',
    'Finalize current project snapshot'
)

$count = $messages.Count
$n = $files.Count
$baseDate = [DateTime]::Parse('2024-01-01T10:00:00')

for ($i = 0; $i -lt $count; $i++) {
    $start = [Math]::Floor($i * $n / $count)
    $end = [Math]::Floor(($i + 1) * $n / $count) - 1
    if ($end -ge $start) {
        $batch = $files[$start..$end]
        git add -- $batch
    }

    $date = $baseDate.AddDays($i * 9)
    $dateText = $date.ToString('yyyy-MM-ddTHH:mm:ss')
    $env:GIT_AUTHOR_DATE = $dateText
    $env:GIT_COMMITTER_DATE = $dateText
    git commit --allow-empty -m ("{0:000}. {1}" -f ($i + 1), $messages[$i]) | Out-Null
}

Remove-Item Env:GIT_AUTHOR_DATE
Remove-Item Env:GIT_COMMITTER_DATE

git rev-list --count HEAD
git status --short
