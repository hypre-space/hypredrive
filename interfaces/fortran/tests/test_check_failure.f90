program test_check_failure
   use, intrinsic :: iso_c_binding
   use hypredrive
   implicit none

   call HYPREDRV_Check(HYPREDRV_ErrorInvalidValue('intentional Fortran check failure'//c_null_char))
end program test_check_failure
