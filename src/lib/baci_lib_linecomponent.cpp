/*----------------------------------------------------------------------*/
/*! \file
\brief Various components that make up an input line
\level 0
*/
/*----------------------------------------------------------------------*/

#include "baci_lib_linecomponent.H"

#include "baci_lib_utils_cond_and_mat_definition.H"

#include <cstddef>
#include <iterator>
#include <utility>

namespace INPUT
{

  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  LineComponent::LineComponent(std::string name, bool optional)
      : optional_(optional), name_(std::move(name))
  {
  }


  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  SeparatorComponent::SeparatorComponent(
      std::string separator, std::string description, bool optional)
      : ::INPUT::LineComponent("*SEPARATOR*", optional),
        separator_(std::move(separator)),
        description_(std::move(description))
  {
  }


  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  void SeparatorComponent::DefaultLine(std::ostream& stream) { stream << separator_; }


  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  void SeparatorComponent::Print(std::ostream& stream, const DRT::Container& cond)
  {
    stream << separator_;
  }

  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  void SeparatorComponent::Describe(std::ostream& stream)
  {
    stream << "    " << std::setw(15) << std::left << separator_ << std::setw(15) << std::left
           << (optional_ ? "(optional)" : "") << description_;
  }

  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  std::string SeparatorComponent::WriteReadTheDocs() { return separator_; }

  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  std::vector<std::string> SeparatorComponent::WriteReadTheDocsTableRow() const
  {
    std::vector<std::string> tablerow;

    tablerow.push_back(separator_);
    tablerow.emplace_back((optional_ ? "yes" : ""));
    std::string descriptionstring = "";
    tablerow.push_back(description_);
    return tablerow;
  }


  /*-----------------------------------------------------------------------------*
   *-----------------------------------------------------------------------------*/
  Teuchos::RCP<std::stringstream> SeparatorComponent::Read(const std::string& section_name,
      Teuchos::RCP<std::stringstream> condline, DRT::Container& line)
  {
    // try to find line parameter label "separator_" (with leading and trailing white spaces for
    // uniqueness) in stringstream "condline"
    size_t position = condline->str().find(" " + separator_ + " ");

    // case: line parameter label "separator_" not found
    if (position == std::string::npos)
    {
      if (optional_)
        // move stringstream position to end of "condline" in case of optional line parameter
        condline->seekg(0, condline->end);
      else
        // return error in case a required line parameter is not specified
        dserror("Required parameter '%s' for section '%s' not specified in input file!",
            separator_.c_str(), section_name.c_str());
    }
    // case: found line parameter label "separator_"
    else
    {
      // care for leading white space in search string ("position" should indicate position of first
      // actual character of parameter label)
      position++;

      // remove line parameter label "separator_" from stringstream "condline"
      condline->str(condline->str().erase(position, separator_.size()));

      // set current position in stringstream "condline" in front of value associated with line
      // parameter label "separator_"
      condline->seekg((std::streampos)position);
    }

    return condline;
  }


  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  StringComponent::StringComponent(std::string name, std::string defaultvalue, bool optional)
      : ::INPUT::LineComponent(std::move(name), optional), defaultvalue_(std::move(defaultvalue))
  {
  }


  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  void StringComponent::DefaultLine(std::ostream& stream) { stream << defaultvalue_; }


  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  void StringComponent::Print(std::ostream& stream, const DRT::Container& cond)
  {
    stream << *cond.Get<std::string>(Name());
  }


  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  void StringComponent::Describe(std::ostream& stream) {}


  /*---------------------------------------------------------------------------*
   *---------------------------------------------------------------------------*/
  Teuchos::RCP<std::stringstream> StringComponent::Read(const std::string& section_name,
      Teuchos::RCP<std::stringstream> condline, DRT::Container& line)
  {
    // initialize string parameter value to be read
    std::string str = defaultvalue_;

    // get current position in stringstream "condline"
    std::streampos position = condline->tellg();

    // only try to read string parameter value in case the associated parameter label appears in
    // line line of input file
    if ((size_t)position != condline->str().size())
    {
      // extract string parameter value from stringstream "condline"
      *condline >> str;

      // return error in case the extraction was not successful
      if (str.empty())
        dserror("Value of parameter '%s' for section '%s' not properly specified in input file!",
            Name().c_str(), section_name.c_str());

      // remove string parameter value from stringstream "condline"
      condline->str(condline->str().erase((size_t)condline->tellg() - str.size(), str.size()));

      // reset current position in stringstream "condline"
      condline->seekg(position);
    }

    // add double parameter value to line parameter list
    line.Add(Name(), str);

    return condline;
  }



  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  SelectionComponent::SelectionComponent(std::string name, std::string defaultvalue,
      const Teuchos::Array<std::string>& datfilevalues,
      const Teuchos::Array<std::string>& stringcondvalues, bool optional)
      : ::INPUT::LineComponent(std::move(name)),
        defaultvalue_(std::move(defaultvalue)),
        datfilevalues_(datfilevalues),
        stringcondvalues_(stringcondvalues),
        intcondvalues_(Teuchos::tuple<int>(-1)),
        stringtostring_(true)
  {
    if (std::find(datfilevalues_.begin(), datfilevalues_.end(), defaultvalue_) ==
        datfilevalues_.end())
    {
      dserror("Invalid default value '%s'.", defaultvalue_.c_str());
    }
    if (datfilevalues_.size() != stringcondvalues_.size())
    {
      dserror("Input file values must match condition values.");
    }
  }


  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  SelectionComponent::SelectionComponent(std::string name, std::string defaultvalue,
      const Teuchos::Array<std::string>& datfilevalues, const Teuchos::Array<int>& intcondvalues,
      bool optional)
      : ::INPUT::LineComponent(std::move(name)),
        defaultvalue_(std::move(defaultvalue)),
        datfilevalues_(datfilevalues),
        stringcondvalues_(Teuchos::tuple<std::string>("notdefined")),
        intcondvalues_(intcondvalues),
        stringtostring_(false)
  {
    if (std::find(datfilevalues_.begin(), datfilevalues_.end(), defaultvalue_) ==
        datfilevalues_.end())
    {
      dserror("Invalid default value '%s'.", defaultvalue_.c_str());
    }
    if (datfilevalues_.size() != intcondvalues_.size())
    {
      dserror("Input file values must match condition values.");
    }
  }

  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  void SelectionComponent::DefaultLine(std::ostream& stream) { stream << defaultvalue_; }

  std::string SelectionComponent::WriteReadTheDocs() { return "<" + Name() + ">"; }


  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  Teuchos::Array<std::string> SelectionComponent::GetOptions() { return datfilevalues_; }


  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  void SelectionComponent::Print(std::ostream& stream, const DRT::Container& cond)
  {
    stream << *cond.Get<std::string>(Name());
  }


  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  Teuchos::RCP<std::stringstream> SelectionComponent::Read(const std::string& section_name,
      Teuchos::RCP<std::stringstream> condline, DRT::Container& condition)
  {
    std::size_t position{};
    std::string selected_value{defaultvalue_};
    for (unsigned i = 0; i < datfilevalues_.size(); ++i)
    {
      position = condline->str().find(" " + datfilevalues_[i] + " ");
      if (position != std::string::npos)
      {
        selected_value = datfilevalues_[i];
        break;
      }
    }

    {
      // care for leading white space in search string ("position" should indicate position of first
      // actual character of parameter label)
      position++;

      // Is this necessary?
      condline->str(condline->str().erase(position, selected_value.size()));

      // set current position in stringstream "condline" in front of value associated with line
      // parameter label "separator_"
      condline->seekg((std::streampos)position);
    }

    auto i = std::find(datfilevalues_.begin(), datfilevalues_.end(), selected_value);
    const unsigned pos = std::distance(datfilevalues_.begin(), i);
    // choose, if we have an array based on std::string or int
    if (stringtostring_)
      condition.Add(Name(), stringcondvalues_[pos]);
    else
      condition.Add(Name(), intcondvalues_[pos]);

    return condline;
  }



  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  IntComponent::IntComponent(std::string name, IntComponentData data)
      : ::INPUT::LineComponent(std::move(name), data.optional), data_(data)
  {
  }


  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  void IntComponent::DefaultLine(std::ostream& stream)
  {
    if (data_.none_allowed)
      stream << "none";
    else
      stream << data_.default_value;
  }


  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  void IntComponent::Print(std::ostream& stream, const DRT::Container& cond)
  {
    int n = cond.GetInt(Name());
    if (data_.none_allowed and n == -1)
      stream << "none ";
    else
    {
      if (data_.fortran_style) n += 1;
      stream << n;
    }
  }


  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  std::string IntComponent::WriteReadTheDocs()
  {
    return data_.none_allowed ? "none" : std::to_string(data_.default_value);
  }


  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  void IntComponent::Describe(std::ostream& stream) {}


  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  Teuchos::RCP<std::stringstream> IntComponent::Read(const std::string& section_name,
      Teuchos::RCP<std::stringstream> condline, DRT::Container& condition)
  {
    // initialize integer parameter value to be read
    int nnumber = data_.default_value;

    // get current position in stringstream "condline"
    std::streampos position = condline->tellg();

    // only try to read integer parameter value in case the associated parameter label appears in
    // line line of input file
    if ((size_t)position != condline->str().size())
    {
      // extract integer vector component as string
      std::string snumber;
      *condline >> snumber;
      if (!optional_ or !snumber.empty())
      {
        // in case 'none' is allowed as an input value
        if ((data_.none_allowed and snumber == "none"))
        {
          nnumber = -1;
        }
        // all other cases
        else
        {
          nnumber = DRT::UTILS::ConvertAndValidateStringToNumber<int>(
              snumber, Name(), section_name, 1, optional_);
        }
      }
      if (data_.fortran_style)
      {
        if (not data_.none_allowed or nnumber != -1) nnumber -= 1;
      }

      // remove parameter value from stringstream "condline"
      condline->str(
          condline->str().erase((size_t)condline->tellg() - snumber.size(), snumber.size()));

      // reset current position in stringstream "condline"
      condline->seekg(position);
    }

    // add int parameter value to line parameter list
    condition.Add(Name(), nnumber);

    return condline;
  }


  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  IntVectorComponent::IntVectorComponent(std::string name, int length, IntComponentData data)
      : ::INPUT::LineComponent(std::move(name), data.optional), length_(length), data_(data)
  {
  }



  IntVectorComponent::IntVectorComponent(
      std::string name, LengthDefinition length_from_component, IntComponentData data)
      : ::INPUT::LineComponent(std::move(name)),
        length_(std::move(length_from_component)),
        data_(data)
  {
  }


  namespace
  {
    struct DefaultLengthVisitor
    {
      int operator()(int length) { return length; }
      int operator()(const LengthDefinition& length) { return 1; }
    };
  }  // namespace


  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  void IntVectorComponent::DefaultLine(std::ostream& stream)
  {
    using namespace std::string_literals;
    const int default_length = std::visit(DefaultLengthVisitor{}, length_);
    const std::string default_value = std::invoke(
        [&]()
        {
          if (data_.none_allowed) return "none "s;
          if (data_.fortran_style)
            return "-1 "s;
          else
            return std::to_string(data_.default_value) + " ";
        });

    for (int i = 0; i < default_length; ++i) stream << default_value;
  }


  std::string IntVectorComponent::WriteReadTheDocs()
  {
    std::string parameterstring = "<int vec";
    if (data_.none_allowed) parameterstring += " [incl none]";
    parameterstring += ":" + Name() + "> ";
    return parameterstring;
  }

  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  void IntVectorComponent::Print(std::ostream& stream, const DRT::Container& cond)
  {
    const auto* v = cond.Get<std::vector<int>>(Name());
    for (int i : *v)
    {
      if (data_.none_allowed and i == -1)
        stream << "none ";
      else
        stream << i + 1 << " ";
    }
  }


  namespace
  {
    struct LengthVisitor
    {
      int operator()(int length) { return length; }
      int operator()(const LengthDefinition& length) { return length(condition); }

      const DRT::Container& condition;
    };
  }  // namespace
     //

  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  Teuchos::RCP<std::stringstream> IntVectorComponent::Read(const std::string& section_name,
      Teuchos::RCP<std::stringstream> condline, DRT::Container& condition)
  {
    const int initialize_value = data_.default_value + (data_.fortran_style ? -1 : 0);
    const int dynamic_length = std::visit(LengthVisitor{condition}, length_);
    // initialize integer parameter vector to be read
    std::vector<int> nnumbers(dynamic_length, initialize_value);

    // get current position in stringstream "condline"
    std::streampos position = condline->tellg();

    // only try to read integer parameter vector in case the associated parameter label appears in
    // line line of input file
    if ((size_t)position != condline->str().size())
    {
      // extract integer parameter vector from stringstream "condline"
      for (auto& current_nnumber : nnumbers)
      {
        // extract integer vector component as string
        std::string snumber;
        *condline >> snumber;

        // in case 'none' is allowed as an input value
        if (data_.none_allowed and snumber == "none")
        {
          current_nnumber = -1;
        }
        // in case the parameter is optional and no value is given
        else if (optional_ and snumber.empty())
        {
          break;
        }
        // all other cases
        else
        {
          current_nnumber = DRT::UTILS::ConvertAndValidateStringToNumber<int>(
              snumber, Name(), section_name, dynamic_length, optional_);
        }

        if (data_.fortran_style)
        {
          if (not data_.none_allowed or current_nnumber != -1) current_nnumber -= 1;
        }

        // remove parameter value from stringstream "condline"
        condline->str(
            condline->str().erase((size_t)condline->tellg() - snumber.size(), snumber.size()));

        // reset current position in stringstream "condline"
        condline->seekg(position);
      }
    }

    // add int parameter vector to line parameter list
    condition.Add(Name(), nnumbers);

    return condline;
  }

  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  void IntVectorComponent::SetLength(int length) { length_ = length; }



  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  void IntVectorComponent::Describe(std::ostream& stream) {}


  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  RealComponent::RealComponent(std::string name, RealComponentData data)
      : ::INPUT::LineComponent(std::move(name), data.optional), data_(data)
  {
  }


  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  void RealComponent::DefaultLine(std::ostream& stream) { stream << data_.default_value; }


  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  void RealComponent::Print(std::ostream& stream, const DRT::Container& cond)
  {
    stream << cond.GetDouble(Name());
  }


  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  void RealComponent::Describe(std::ostream& stream) {}


  /*---------------------------------------------------------------------------*
   | Read double parameter value from line line of input file   fang 08/14 |
   *---------------------------------------------------------------------------*/
  Teuchos::RCP<std::stringstream> RealComponent::Read(const std::string& section_name,
      Teuchos::RCP<std::stringstream> condline, DRT::Container& line)
  {
    // initialize double parameter value to be read
    double nnumber = data_.default_value;

    // get current position in stringstream "condline"
    std::streampos position = condline->tellg();

    // only try to read double parameter value in case the associated parameter label appears in
    // line line of input file
    if ((size_t)position != condline->str().size())
    {
      // read string from stream (need to handle doubles and "none" arguments)
      std::string snumber;
      *condline >> snumber;

      if (!(optional_ && snumber.empty()))
      {
        nnumber = DRT::UTILS::ConvertAndValidateStringToNumber<double>(
            snumber, Name(), section_name, 1, optional_);

        // remove parameter value from stringstream "condline"
        condline->str(
            condline->str().erase((size_t)condline->tellg() - snumber.size(), snumber.size()));

        // reset current position in stringstream "condline"
        condline->seekg(position);
      }
    }

    // add double parameter value to line parameter list
    line.Add(Name(), nnumber);

    return condline;
  }



  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  RealVectorComponent::RealVectorComponent(std::string name, int length, RealComponentData data)
      : ::INPUT::LineComponent(std::move(name), data.optional), length_(length), data_(data)
  {
  }


  RealVectorComponent::RealVectorComponent(
      std::string name, LengthDefinition length, RealComponentData data)
      : ::INPUT::LineComponent(std::move(name), data.optional),
        length_(std::move(length)),
        data_(data)
  {
  }


  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  void RealVectorComponent::DefaultLine(std::ostream& stream)
  {
    const int default_length = std::visit(DefaultLengthVisitor{}, length_);
    for (int i = 0; i < default_length; ++i) stream << data_.default_value;
  }



  std::string RealVectorComponent::WriteReadTheDocs() { return "<real vec:" + Name() + "> "; }


  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  void RealVectorComponent::Print(std::ostream& stream, const DRT::Container& cond)
  {
    const auto* v = cond.Get<std::vector<double>>(Name());
    for (double i : *v) stream << i << " ";
  }


  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  void RealVectorComponent::Describe(std::ostream& stream) {}


  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  Teuchos::RCP<std::stringstream> RealVectorComponent::Read(const std::string& section_name,
      Teuchos::RCP<std::stringstream> condline, DRT::Container& condition)
  {
    const int dynamic_length = std::visit(LengthVisitor{condition}, length_);
    std::vector<double> nnumbers(dynamic_length, data_.default_value);

    // get current position in stringstream "condline"
    std::streampos position = condline->tellg();

    // only try to read double parameter vector in case the associated parameter label appears in
    // line line of input file
    if ((size_t)position != condline->str().size())
    {
      // extract double parameter vector from stringstream "condline"
      for (auto& current_nnumber : nnumbers)
      {
        // extract double vector component as string
        std::string snumber;
        *condline >> snumber;

        // in case the parameter is optional and no value is
        // given
        if (optional_ and snumber.empty())
        {
          break;
        }
        // all other cases
        else
        {
          current_nnumber = DRT::UTILS::ConvertAndValidateStringToNumber<double>(
              snumber, Name(), section_name, dynamic_length, optional_);
        }

        // remove parameter value from stringstream "condline"
        condline->str(
            condline->str().erase((size_t)condline->tellg() - snumber.size(), snumber.size()));

        // reset current position in stringstream "condline"
        condline->seekg(position);
      }
    }

    // add double parameter vector to line parameter list
    condition.Add(Name(), nnumbers);

    return condline;
  }

  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  void RealVectorComponent::SetLength(int length) { length_ = length; }



  /*======================================================================*/
  /*======================================================================*/
  const std::string BoolComponent::lineTrue_ = "Yes";
  const std::string BoolComponent::lineFalse_ = "No";
  BoolComponent::BoolComponent(std::string name, const bool defaultvalue, bool optional)
      : ::INPUT::LineComponent(std::move(name), optional), defaultvalue_(defaultvalue)
  {
  }

  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  void BoolComponent::DefaultLine(std::ostream& stream) { PrintYesNo(stream, defaultvalue_); }


  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  void BoolComponent::Print(std::ostream& stream, const DRT::Container& cond)
  {
    const bool value = (bool)cond.GetInt(Name());
    PrintYesNo(stream, value);
  }

  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  void BoolComponent::PrintYesNo(std::ostream& stream, const bool value) const
  {
    if (value)
      stream << lineTrue_;
    else
      stream << lineFalse_;
  }

  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  void BoolComponent::Describe(std::ostream& stream) {}


  /*----------------------------------------------------------------------------*
   | Read boolean parameter value from line line of input file   fang 08/14 |
   *----------------------------------------------------------------------------*/
  Teuchos::RCP<std::stringstream> BoolComponent::Read(const std::string& section_name,
      Teuchos::RCP<std::stringstream> condline, DRT::Container& line)
  {
    // initialize boolean parameter value to be read
    bool boolean = defaultvalue_;

    // get current position in stringstream "condline"
    std::streampos position = condline->tellg();

    // only try to read boolean parameter value in case the associated parameter label appears in
    // line line of input file
    if ((size_t)position != condline->str().size())
    {
      // extract boolean parameter value from stringstream "condline" as string
      std::string sboolean;
      *condline >> sboolean;

      // try to convert to bool
      if (sboolean == "Yes" or sboolean == "YES" or sboolean == "yes" or sboolean == "True" or
          sboolean == "TRUE" or sboolean == "true")
        boolean = true;
      else if (sboolean == "No" or sboolean == "NO" or sboolean == "no" or sboolean == "False" or
               sboolean == "FALSE" or sboolean == "false")
        boolean = false;
      else
      {
        // return error in case the conversion was not successful
        dserror("Value of parameter '%s' for section '%s' not properly specified in input file!",
            Name().c_str(), section_name.c_str());
      }

      // remove boolean parameter value from stringstream "condline"
      condline->str(
          condline->str().erase((size_t)condline->tellg() - sboolean.size(), sboolean.size()));

      // reset current position in stringstream "condline"
      condline->seekg(position);
    }

    // add boolean parameter value to line parameter list
    line.Add(Name(), boolean);

    return condline;
  }



  SwitchComponent::SwitchComponent(std::string name, const KeyType& default_key,
      std::map<KeyType, std::pair<std::string, std::vector<Teuchos::RCP<::INPUT::LineComponent>>>>
          choices)
      : ::INPUT::LineComponent(std::move(name)),
        default_key_(default_key),
        choices_(std::move(choices))
  {
    Teuchos::Array<int> keys;
    keys.reserve(choices_.size());
    Teuchos::Array<std::string> names_for_keys;
    names_for_keys.reserve(choices_.size());

    for (const auto& [key, choice] : choices_)
    {
      keys.push_back(key);
      names_for_keys.push_back(choice.first);
    }

    component_for_key_ = std::make_unique<::INPUT::SelectionComponent>(
        Name(), choices_[default_key_].first, names_for_keys, keys);
  }



  void SwitchComponent::DefaultLine(std::ostream& stream)
  {
    component_for_key_->DefaultLine(stream);
    stream << " ";

    for (const auto& component : choices_[default_key_].second)
    {
      component->DefaultLine(stream);
      stream << " ";
    }
  }



  std::vector<std::string> SwitchComponent::WriteReadTheDocsLines()
  {
    std::vector<std::string> all_choices_as_rtd;
    std::transform(choices_.begin(), choices_.end(), std::back_inserter(all_choices_as_rtd),
        [this](const auto& key_components)
        {
          const auto& [key, components] = key_components;

          std::stringstream stream;
          stream << choices_[key].first << " ";
          for (const auto& c : components.second) stream << c->WriteReadTheDocs() << " ";
          return stream.str();
        });

    return all_choices_as_rtd;
  }


  std::string SwitchComponent::WriteReadTheDocs()
  {
    return component_for_key_->WriteReadTheDocs() + " [further parameters]";
  }



  Teuchos::Array<std::string> SwitchComponent::GetOptions()
  {
    return component_for_key_->GetOptions();
  }



  void SwitchComponent::Print(std::ostream& stream, const DRT::Container& cond)
  {
    component_for_key_->Print(stream, cond);
    stream << " ";

    const KeyType selected_key = static_cast<KeyType>(cond.GetInt(component_for_key_->Name()));

    dsassert(choices_.count(selected_key) == 1, "Internal error.");
    for (const auto& component : choices_[selected_key].second)
    {
      component->Print(stream, cond);
      stream << " ";
    }
  }



  Teuchos::RCP<std::stringstream> SwitchComponent::Read(const std::string& section_name,
      Teuchos::RCP<std::stringstream> condline, DRT::Container& condition)
  {
    component_for_key_->Read(section_name, condline, condition);
    const KeyType key = static_cast<KeyType>(condition.GetInt(component_for_key_->Name()));

    dsassert(choices_.count(key) == 1, "Internal error.");

    for (const auto& component : choices_[key].second)
    {
      component->Read(section_name, condline, condition);
    }

    return condline;
  }
}  // namespace INPUT