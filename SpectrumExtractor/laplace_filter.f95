!------------------------------------------------------------------
! This is an f95 library of negative 2D laplace filtering functions
! Last updated : 20210405: JPN
!------------------------------------------------------------------

SUBROUTINE nlaplace_filter(A,A_I,A_J,A_OUT)
! Negative 2D lapace filter on full array.
! Simillar in behaviour to -1*scipy.ndimage.laplace, but slower for large arrays.
!  0  -1   0
! -1   4  -1
!  0  -1   0
!
! Input:
!     A : 2D array with shape (A_I,A_J)
! Output:
!     A_OUT: 2D array with shape (A_I,A_J)
!           It is the negative of the laplace filtered input A array.
!
  IMPLICIT NONE
  INTEGER, intent(in)  :: A_I, A_J
  INTEGER              :: I, J, In, Ip, Jn, Jp
  REAL(8), intent(in), dimension(A_I,A_J) :: A
  REAL(8), intent(out), dimension(A_I,A_J) :: A_OUT

  DO I = 1, A_I
     In = max(1,I-1)
     Ip = min(A_I,I+1)
     DO J = 1, A_J
        Jn = max(1,J-1)
        Jp = min(A_J,J+1)
        A_OUT(I,J) = 4*A(I,J) - A(In,J) - A(Ip,J) - A(I,Jn) - A(I,Jp)
     END DO
  END DO
END SUBROUTINE nlaplace_filter


SUBROUTINE nlaplace_filter_super(A,A_I,A_J,A_OUT)
! This is an optimised routine for effectively doing the following steps to entire array
! It is inspired by the idea in Dokkum's LACosmic paper: http://www.astro.yale.edu/dokkum/lacosmic/
!
! 1) Super sample the input image array by a factor of 2
! 2) Negative 2D lapace filter on this super sampled full array.
! 3) Set any negative values to zero
! 4) And finally rebin back to original size by averaging
!
! In the actual implementation here we do not do the super sampling,
! Instead we have an updated formula which effectively outputs the same output.
!
! Input:
!     A : 2D array with shape (A_I,A_J)
! Output:
!     A_OUT: 2D array with shape (A_I,A_J)
!           It is the negative of the laplace filtered input A array.
!
  IMPLICIT NONE
  INTEGER, intent(in)  :: A_I, A_J
  INTEGER              :: I, J, In, Ip, Jn, Jp
  REAL(8), intent(in), dimension(A_I,A_J) :: A
  REAL(8), intent(out), dimension(A_I,A_J) :: A_OUT

  DO I = 1, A_I
     In = max(1,I-1)
     Ip = min(A_I,I+1)
     DO J = 1, A_J
        Jn = max(1,J-1)
        Jp = min(A_J,J+1)
        ! Effective formula for faster calculation of the steps mentioned in the docstring
        A_OUT(I,J) = (max(0.,2*A(I,J)-A(I,Jn)-A(In,J)) +max(0.,2*A(I,J)-A(In,J)-A(I,Jp)) &
                     +max(0.,2*A(I,J)-A(I,Jp)-A(Ip,J)) +max(0.,2*A(I,J)-A(Ip,J)-A(I,Jn)))/4.
     END DO
  END DO
END SUBROUTINE nlaplace_filter_super


SUBROUTINE nlaplace_filter_intrace(A,A_I,A_J,LowerCoord,UpperCoord,A_OUT)
! Negative 2D lapace filter on the selected pixels between LowerCoord and UpperCoord coordinates
! Simillar in behaviour to -1*scipy.ndimage.laplace, but only for pixels inside the range
!  0  -1   0
! -1   4  -1
!  0  -1   0
!
! Input:
!     A : 2D array with shape (A_I,A_J)
!     LowerCoord: 1D array with length(A_J)
!                 The lower coordinates of the pixel regions in each column we want to process
!     UpperCoord: 1D array with length(A_J)
!                 The upper coordinates of the pixel regions in each column we want to process
! Output:
!     A_OUT: 2D array with shape (A_I,A_J)
!           It is the negative of the laplace filtered values of pixels inside the
!           region of the input A array.
!
  IMPLICIT NONE
  INTEGER, intent(in)  :: A_I, A_J
  INTEGER              :: I, J, In, Ip, Jn, Jp
  REAL(8), intent(in), dimension(A_I,A_J) :: A
  INTEGER, intent(in), dimension(A_J) :: LowerCoord, UpperCoord
  REAL(8), intent(out), dimension(A_I,A_J) :: A_OUT

  DO J = 1, A_J
     Jn = max(1,J-1)
     Jp = min(A_J,J+1)
     ! add 1 to the python coordinate to match index in fortran
     DO I = LowerCoord(J)+1, UpperCoord(J)+1
        In = max(1,I-1)
        Ip = min(A_I,I+1)
        A_OUT(I,J) = 4*A(I,J) - A(In,J) - A(Ip,J) - A(I,Jn) - A(I,Jp)
     END DO
  END DO
END SUBROUTINE nlaplace_filter_intrace


SUBROUTINE nlaplace_filter_intrace_super(A,A_I,A_J,LowerCoord,UpperCoord,A_OUT)
! This is an optimised routine for effectively doing the following steps to
! the selected pixels between LowerCoord and UpperCoord coordinates
! It is inspired by the idea in Dokkum's LACosmic paper: http://www.astro.yale.edu/dokkum/lacosmic/
!
! 1) Super sample the input image array by a factor of 2
! 2) Negative 2D lapace filter on this super sampled full array.
! 3) Set any negative values to zero
! 4) And finally rebin back to original size by averaging
!
! In the actual implementation here we do not do the super sampling,
! Instead we have an updated formula which effectively outputs the same output.
!
! Input:
!     A : 2D array with shape (A_I,A_J)
!     LowerCoord: 1D array with length(A_J)
!                 The lower coordinates of the pixel regions in each column we want to process
!     UpperCoord: 1D array with length(A_J)
!                 The upper coordinates of the pixel regions in each column we want to process
! Output:
!     A_OUT: 2D array with shape (A_I,A_J)
!           It is the negative of the laplace filtered values of pixels inside the
!           region of the input A array.
!
  IMPLICIT NONE
  INTEGER, intent(in)  :: A_I, A_J
  INTEGER              :: I, J, In, Ip, Jn, Jp
  REAL(8), intent(in), dimension(A_I,A_J) :: A
  INTEGER, intent(in), dimension(A_J) :: LowerCoord, UpperCoord
  REAL(8), intent(out), dimension(A_I,A_J) :: A_OUT

  DO J = 1, A_J
     Jn = max(1,J-1)
     Jp = min(A_J,J+1)
     ! add 1 to the python coordinate to match index in fortran
     DO I = LowerCoord(J)+1, UpperCoord(J)+1
        In = max(1,I-1)
        Ip = min(A_I,I+1)
        ! Effective formula for faster calculation of the steps mentioned in the docstring
        A_OUT(I,J) = (max(0.,2*A(I,J)-A(I,Jn)-A(In,J)) +max(0.,2*A(I,J)-A(In,J)-A(I,Jp)) &
                     +max(0.,2*A(I,J)-A(I,Jp)-A(Ip,J)) +max(0.,2*A(I,J)-A(Ip,J)-A(I,Jn)))/4.
     END DO
  END DO
END SUBROUTINE nlaplace_filter_intrace_super
