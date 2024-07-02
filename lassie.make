# GNU Make project makefile autogenerated by Premake
ifndef config
  config=debug
endif

ifndef verbose
  SILENT = @
endif

ifndef CC
  CC = gcc
endif

ifndef CXX
  CXX = g++
endif

ifndef AR
  AR = ar
endif

ifeq ($(config),debug)
  OBJDIR     = obj/Debug/lassie
  TARGETDIR  = .
  TARGET     = $(TARGETDIR)/lassie
  DEFINES   += 
  INCLUDES  += 
  CPPFLAGS  += -MMD -MP $(DEFINES) $(INCLUDES)
  CFLAGS    += $(CPPFLAGS) $(ARCH) -g `pkg-config --cflags gtkmm-2.4` -Wno-deprecated -gstabs -std=c++11
  CXXFLAGS  += $(CFLAGS) 
  LDFLAGS   += `pkg-config --libs --cflags gtkmm-2.4` -Wno-deprecated -lxerces-c -L/usr/local/lib -Llib
  LIBS      += -llcmod -llass -lparser -lpthread -lsndfile
  RESFLAGS  += $(DEFINES) $(INCLUDES) 
  LDDEPS    += lib/liblcmod.a lib/liblass.a lib/libparser.a
  LINKCMD    = $(CXX) -o $(TARGET) $(OBJECTS) $(LDFLAGS) $(RESOURCES) $(ARCH) $(LIBS)
  define PREBUILDCMDS
  endef
  define PRELINKCMDS
  endef
  define POSTBUILDCMDS
  endef
endif

ifeq ($(config),release)
  OBJDIR     = obj/Release/lassie
  TARGETDIR  = .
  TARGET     = $(TARGETDIR)/lassie
  DEFINES   += 
  INCLUDES  += 
  CPPFLAGS  += -MMD -MP $(DEFINES) $(INCLUDES)
  CFLAGS    += $(CPPFLAGS) $(ARCH) -O2 `pkg-config --cflags gtkmm-2.4` -Wno-deprecated -gstabs -std=c++11
  CXXFLAGS  += $(CFLAGS) 
  LDFLAGS   += -s `pkg-config --libs --cflags gtkmm-2.4` -Wno-deprecated -lxerces-c -L/usr/local/lib -Llib
  LIBS      += -llcmod -llass -lparser -lpthread -lsndfile
  RESFLAGS  += $(DEFINES) $(INCLUDES) 
  LDDEPS    += lib/liblcmod.a lib/liblass.a lib/libparser.a
  LINKCMD    = $(CXX) -o $(TARGET) $(OBJECTS) $(LDFLAGS) $(RESOURCES) $(ARCH) $(LIBS)
  define PREBUILDCMDS
  endef
  define PRELINKCMDS
  endef
  define POSTBUILDCMDS
  endef
endif

OBJECTS := \
	$(OBJDIR)/IEvent.o \
	$(OBJDIR)/PaletteViewController.o \
	$(OBJDIR)/FileOperations.o \
	$(OBJDIR)/MainWindow.o \
	$(OBJDIR)/HelpOperations.o \
	$(OBJDIR)/ProjectViewController.o \
	$(OBJDIR)/ObjectWindow.o \
	$(OBJDIR)/EnvLibDrawingArea.o \
	$(OBJDIR)/Main.o \
	$(OBJDIR)/EventAttributesViewController.o \
	$(OBJDIR)/FunctionGenerator.o \
	$(OBJDIR)/EnvelopeLibraryEntry.o \
	$(OBJDIR)/PartialWindow.o \
	$(OBJDIR)/EnvelopeLibraryWindow.o \
	$(OBJDIR)/MarkovModelLibraryWindow.o \
	$(OBJDIR)/SharedPointers.o \

RESOURCES := \

SHELLTYPE := msdos
ifeq (,$(ComSpec)$(COMSPEC))
  SHELLTYPE := posix
endif
ifeq (/bin,$(findstring /bin,$(SHELL)))
  SHELLTYPE := posix
endif

.PHONY: clean prebuild prelink

all: $(TARGETDIR) $(OBJDIR) prebuild prelink $(TARGET)
	@:

$(TARGET): $(GCH) $(OBJECTS) $(LDDEPS) $(RESOURCES)
	@echo Linking lassie
	$(SILENT) $(LINKCMD)
	$(POSTBUILDCMDS)

$(TARGETDIR):
	@echo Creating $(TARGETDIR)
ifeq (posix,$(SHELLTYPE))
	$(SILENT) mkdir -p $(TARGETDIR)
else
	$(SILENT) mkdir $(subst /,\\,$(TARGETDIR))
endif

$(OBJDIR):
	@echo Creating $(OBJDIR)
ifeq (posix,$(SHELLTYPE))
	$(SILENT) mkdir -p $(OBJDIR)
else
	$(SILENT) mkdir $(subst /,\\,$(OBJDIR))
endif

clean:
	@echo Cleaning lassie
ifeq (posix,$(SHELLTYPE))
	$(SILENT) rm -f  $(TARGET)
	$(SILENT) rm -rf $(OBJDIR)
else
	$(SILENT) if exist $(subst /,\\,$(TARGET)) del $(subst /,\\,$(TARGET))
	$(SILENT) if exist $(subst /,\\,$(OBJDIR)) rmdir /s /q $(subst /,\\,$(OBJDIR))
endif

prebuild:
	$(PREBUILDCMDS)

prelink:
	$(PRELINKCMDS)

ifneq (,$(PCH))
$(GCH): $(PCH)
	@echo $(notdir $<)
	-$(SILENT) cp $< $(OBJDIR)
	$(SILENT) $(CXX) $(CXXFLAGS) -o "$@" -c "$<"
endif

$(OBJDIR)/IEvent.o: LASSIE/src/IEvent.cpp
	@echo $(notdir $<)
	$(SILENT) $(CXX) $(CXXFLAGS) -o "$@" -c "$<"
$(OBJDIR)/PaletteViewController.o: LASSIE/src/PaletteViewController.cpp
	@echo $(notdir $<)
	$(SILENT) $(CXX) $(CXXFLAGS) -o "$@" -c "$<"
$(OBJDIR)/FileOperations.o: LASSIE/src/FileOperations.cpp
	@echo $(notdir $<)
	$(SILENT) $(CXX) $(CXXFLAGS) -o "$@" -c "$<"
$(OBJDIR)/MainWindow.o: LASSIE/src/MainWindow.cpp
	@echo $(notdir $<)
	$(SILENT) $(CXX) $(CXXFLAGS) -o "$@" -c "$<"
$(OBJDIR)/HelpOperations.o: LASSIE/src/HelpOperations.cpp
	@echo $(notdir $<)
	$(SILENT) $(CXX) $(CXXFLAGS) -o "$@" -c "$<"
$(OBJDIR)/ProjectViewController.o: LASSIE/src/ProjectViewController.cpp
	@echo $(notdir $<)
	$(SILENT) $(CXX) $(CXXFLAGS) -o "$@" -c "$<"
$(OBJDIR)/ObjectWindow.o: LASSIE/src/ObjectWindow.cpp
	@echo $(notdir $<)
	$(SILENT) $(CXX) $(CXXFLAGS) -o "$@" -c "$<"
$(OBJDIR)/EnvLibDrawingArea.o: LASSIE/src/EnvLibDrawingArea.cpp
	@echo $(notdir $<)
	$(SILENT) $(CXX) $(CXXFLAGS) -o "$@" -c "$<"
$(OBJDIR)/Main.o: LASSIE/src/Main.cpp
	@echo $(notdir $<)
	$(SILENT) $(CXX) $(CXXFLAGS) -o "$@" -c "$<"
$(OBJDIR)/EventAttributesViewController.o: LASSIE/src/EventAttributesViewController.cpp
	@echo $(notdir $<)
	$(SILENT) $(CXX) $(CXXFLAGS) -o "$@" -c "$<"
$(OBJDIR)/FunctionGenerator.o: LASSIE/src/FunctionGenerator.cpp
	@echo $(notdir $<)
	$(SILENT) $(CXX) $(CXXFLAGS) -o "$@" -c "$<"
$(OBJDIR)/EnvelopeLibraryEntry.o: LASSIE/src/EnvelopeLibraryEntry.cpp
	@echo $(notdir $<)
	$(SILENT) $(CXX) $(CXXFLAGS) -o "$@" -c "$<"
$(OBJDIR)/PartialWindow.o: LASSIE/src/PartialWindow.cpp
	@echo $(notdir $<)
	$(SILENT) $(CXX) $(CXXFLAGS) -o "$@" -c "$<"
$(OBJDIR)/EnvelopeLibraryWindow.o: LASSIE/src/EnvelopeLibraryWindow.cpp
	@echo $(notdir $<)
	$(SILENT) $(CXX) $(CXXFLAGS) -o "$@" -c "$<"
$(OBJDIR)/MarkovModelLibraryWindow.o: LASSIE/src/MarkovModelLibraryWindow.cpp
	@echo $(notdir $<)
	$(SILENT) $(CXX) $(CXXFLAGS) -o "$@" -c "$<"
$(OBJDIR)/SharedPointers.o: LASSIE/src/SharedPointers.cpp
	@echo $(notdir $<)
	$(SILENT) $(CXX) $(CXXFLAGS) -o "$@" -c "$<"

-include $(OBJECTS:%.o=%.d)
