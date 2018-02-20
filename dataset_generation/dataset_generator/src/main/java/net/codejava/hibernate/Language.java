package net.codejava.hibernate;
import javax.persistence.*;

@Entity
@Table(name="language")
public class Language {
	private Long language_id;
	private String language_value;
	
	
	public Language() {
		super();
		// TODO Auto-generated constructor stub
	}

	public Language(Long language_id, String language_value) {
		super();
		this.language_id = language_id;
		this.language_value = language_value;
	}


	@Id
	@Column(name = "language_id")
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	public Long getLanguage_id() {
		return language_id;
	}

	public void setLanguage_id(Long language_id) {
		this.language_id = language_id;
	}

	public String getLanguage_value() {
		return language_value;
	}

	public void setLanguage_value(String language_value) {
		this.language_value = language_value;
	}

	@Override
	public String toString() {
		return "Language [language_id=" + language_id + ", language_value=" + language_value + "]";
	}
}
