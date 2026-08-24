# prompt.md 3.9: compile with warnings enabled; prefer -Wall -Wextra -Wpedantic.
function(sihps_set_warnings target)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
  )
endfunction()
